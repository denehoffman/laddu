use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{LadduDataError, LadduDataResult, schema::Schema};

use super::source_error;

/// Results shared by file-backed source builders.
pub(crate) struct SourceBuild<C> {
    pub(crate) files: Arc<[Arc<PathBuf>]>,
    pub(crate) context: C,
    pub(crate) schema: Arc<Schema>,
}

/// Common source-builder inputs consumed by [`build_source`].
pub(crate) struct SourceBuildOptions<'a> {
    pub(crate) pattern: &'a str,
    pub(crate) sort: bool,
    pub(crate) format: &'a str,
    pub(crate) explicit_schema: Option<Arc<Schema>>,
    pub(crate) infer_schema: bool,
    pub(crate) validate_all_files: bool,
}

/// Resolves and validates a source glob shared by file-backed backends.
///
/// Backends still own metadata inspection, schema inference, and validation;
/// this helper keeps path matching, deterministic ordering, and empty-match
/// behavior identical across them.
pub(crate) fn resolve_source_files(
    pattern: &str,
    sort: bool,
    format: &str,
) -> LadduDataResult<Arc<[Arc<PathBuf>]>> {
    let mut files: Vec<PathBuf> = glob::glob(pattern)
        .map_err(|error| source_error("resolve source glob", pattern, error))?
        .collect::<Result<_, _>>()
        .map_err(|error| source_error("resolve source files", pattern, error))?;

    if sort {
        files.sort();
    }

    if files.is_empty() {
        return Err(source_error(
            "resolve source glob",
            pattern,
            format!("no {format} files matched glob"),
        ));
    }

    Ok(files.into_iter().map(Arc::new).collect())
}

/// Resolves source files, prepares backend context, selects or infers the
/// logical schema, and validates every matched file when requested.
///
/// The backend supplies the context and metadata callbacks. Parquet uses the
/// unit context, while ROOT uses the selected tree name; keeping that context
/// in this orchestration avoids subtly different explicit/inferred and
/// validate-all-files behavior between backends.
pub(crate) fn build_source<C, Prepare, Infer, Validate>(
    options: SourceBuildOptions<'_>,
    prepare: Prepare,
    infer: Infer,
    mut validate: Validate,
) -> LadduDataResult<SourceBuild<C>>
where
    Prepare: FnOnce(&Path) -> LadduDataResult<C>,
    Infer: FnOnce(&Path, &C) -> LadduDataResult<Schema>,
    Validate: FnMut(&Path, &Schema, &C) -> LadduDataResult<()>,
{
    let files = resolve_source_files(options.pattern, options.sort, options.format)?;
    let first_path = files[0].as_ref();
    let context = prepare(first_path)?;

    let schema = match options.explicit_schema {
        Some(schema) => schema,
        None if options.infer_schema => Arc::new(infer(first_path, &context)?),
        None => return Err(LadduDataError::InvalidArgument("schema required")),
    };

    if options.validate_all_files {
        for file in files.iter() {
            validate(file.as_ref(), &schema, &context)?;
        }
    }

    Ok(SourceBuild {
        files,
        context,
        schema,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "laddu-source-builder-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn resolves_sorted_nonempty_globs_and_rejects_empty_matches() {
        let dir = temp_dir();
        let second = dir.join("b.parquet");
        let first = dir.join("a.parquet");
        fs::write(&second, []).unwrap();
        fs::write(&first, []).unwrap();

        let pattern = format!("{}/*.parquet", dir.display());
        let files = resolve_source_files(&pattern, true, "parquet").unwrap();
        assert_eq!(files.as_ref(), &[Arc::new(first), Arc::new(second)]);

        let empty = format!("{}/*.root", dir.display());
        assert!(matches!(
            resolve_source_files(&empty, true, "ROOT"),
            Err(LadduDataError::Source(message)) if message.contains("no ROOT files")
        ));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn builds_explicit_or_inferred_schema_and_validates_all_files() {
        let dir = temp_dir();
        let first = dir.join("a.data");
        let second = dir.join("b.data");
        fs::write(&first, []).unwrap();
        fs::write(&second, []).unwrap();
        let pattern = format!("{}/*.data", dir.display());
        let validations = AtomicUsize::new(0);

        let explicit = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap());
        let built = build_source(
            SourceBuildOptions {
                pattern: &pattern,
                sort: true,
                format: "data",
                explicit_schema: Some(Arc::clone(&explicit)),
                infer_schema: false,
                validate_all_files: true,
            },
            |_| Ok(()),
            |_, _| unreachable!("explicit schema skips inference"),
            |_, schema, _| {
                assert_eq!(schema, explicit.as_ref());
                validations.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(built.schema.as_ref(), explicit.as_ref());
        assert_eq!(built.files.len(), 2);
        assert_eq!(validations.load(Ordering::Relaxed), 2);

        let inferred = build_source(
            SourceBuildOptions {
                pattern: &pattern,
                sort: true,
                format: "data",
                explicit_schema: None,
                infer_schema: true,
                validate_all_files: false,
            },
            |_| Ok(()),
            |_, _| Schema::new(["p"], std::iter::empty::<&str>(), false),
            |_, _, _| unreachable!("validation is disabled"),
        )
        .unwrap();
        assert_eq!(inferred.schema.p4s().len(), 1);

        fs::remove_dir_all(dir).unwrap();
    }
}
