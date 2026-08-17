use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::LadduDataResult;

use super::{WritePlan, sink_error};

/// Resolves a base output path for serial or distributed writes.
#[derive(Clone, Debug)]
pub struct OutputPath {
    base: PathBuf,
    mode: OutputMode,
}

/// Policy for resolving a concrete output path.
#[derive(Clone, Copy, Debug, Default)]
pub enum OutputMode {
    /// Select single-file or per-rank output from the write plan.
    #[default]
    Auto,
    /// Write exactly one file; invalid for distributed writes.
    SingleFile,
    /// Write a rank-specific file.
    PerRankFiles,
}

impl OutputPath {
    /// Creates an automatically resolved output path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            base: path.into(),
            mode: OutputMode::Auto,
        }
    }

    /// Returns this path with an explicit output mode.
    pub fn with_mode(mut self, mode: OutputMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the unresolved base path.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Returns the output mode.
    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Resolves the concrete path for a write plan.
    ///
    /// # Errors
    ///
    /// Returns [`crate::LadduDataError`] when single-file output is requested for a
    /// distributed plan.
    pub fn resolve(&self, plan: WritePlan, default_extension: &str) -> LadduDataResult<PathBuf> {
        let mode = match self.mode {
            OutputMode::Auto if plan.is_distributed() => OutputMode::PerRankFiles,
            OutputMode::Auto => OutputMode::SingleFile,
            mode => mode,
        };

        match mode {
            OutputMode::SingleFile => {
                if plan.is_distributed() {
                    return Err(sink_error(
                        "resolve output path",
                        self.base.display(),
                        "single-file output is unsafe with multiple MPI ranks; use per-rank output",
                    ));
                }

                Ok(self.base.clone())
            }

            OutputMode::PerRankFiles => Ok(per_rank_path(
                &self.base,
                plan.rank(),
                plan.nranks(),
                default_extension,
            )),

            OutputMode::Auto => unreachable!(),
        }
    }

    /// Creates a file's parent directories when absent.
    ///
    /// # Errors
    ///
    /// Returns [`crate::LadduDataError`] when a required directory cannot be created.
    pub fn create_parent_dirs(path: &Path) -> LadduDataResult<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|e| sink_error("create output directory", parent.display(), e))?;
        }

        Ok(())
    }
}

fn per_rank_path(base: &Path, rank: usize, nranks: usize, default_extension: &str) -> PathBuf {
    if base.extension().is_none() {
        let ext = default_extension.trim_start_matches('.');
        return base.join(format!("part-rank{rank:05}-of{nranks:05}.{ext}"));
    }

    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension().unwrap_or_default().to_string_lossy();

    parent.join(format!("{stem}.rank{rank:05}-of{nranks:05}.{ext}"))
}
