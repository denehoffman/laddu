//! Exact, immutable row-data columns independent of expression scalars.

use std::{fmt, str::FromStr, sync::Arc};

use crate::{LadduDataError, LadduDataResult};

macro_rules! integer_columns {
    ($(($variant:ident, $ty:ty, $name:literal, $alias:literal)),* $(,)?) => {
        /// Exact storage dtype for non-expression row data.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub enum ColumnDType {
            $(#[doc = concat!("An exact `", $name, "` column.")]
            $variant,)*
        }

        impl ColumnDType {
            /// Returns the canonical NumPy-style dtype name.
            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $name,)* }
            }

            /// Returns storage bytes per row.
            pub const fn width(self) -> usize {
                match self { $(Self::$variant => std::mem::size_of::<$ty>(),)* }
            }
        }

        impl FromStr for ColumnDType {
            type Err = LadduDataError;
            fn from_str(value: &str) -> LadduDataResult<Self> {
                match value {
                    $($name | $alias => Ok(Self::$variant),)*
                    _ => Err(LadduDataError::Schema(format!("unsupported column dtype: {value}"))),
                }
            }
        }

        /// One exact row-data value.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum ColumnValue {
            $(#[doc = concat!("An exact `", $name, "` value.")]
            $variant($ty),)*
        }

        impl ColumnValue {
            /// Returns the value's exact dtype.
            pub const fn dtype(self) -> ColumnDType {
                match self { $(Self::$variant(_) => ColumnDType::$variant,)* }
            }
        }

        /// An immutable exact row-data column.
        ///
        /// Unlike floating-point expression scalars, these values never pass
        /// through floating-point storage or runtime precision conversion.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum Column {
            $(#[doc = concat!("An immutable `", $name, "` buffer.")]
            $variant(Arc<[$ty]>),)*
        }

        impl Column {
            /// Returns the exact storage dtype.
            pub fn dtype(&self) -> ColumnDType {
                match self { $(Self::$variant(_) => ColumnDType::$variant,)* }
            }

            /// Returns the number of values.
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(values) => values.len(),)* }
            }

            /// Returns one exact value.
            ///
            /// # Panics
            /// Panics when `row` is outside the column.
            pub fn at(&self, row: usize) -> ColumnValue {
                match self { $(Self::$variant(values) => ColumnValue::$variant(values[row]),)* }
            }

            /// Appends exact little-endian values to a transport buffer.
            pub fn append_le_bytes(&self, output: &mut Vec<u8>) {
                match self { $(Self::$variant(values) => {
                    for value in values.iter() { output.extend_from_slice(&value.to_le_bytes()); }
                },)* }
            }

            /// Decodes a buffer of exact little-endian values.
            ///
            /// # Errors
            /// Returns an error if the byte count is not a multiple of the dtype width.
            pub fn from_le_bytes(dtype: ColumnDType, bytes: &[u8]) -> LadduDataResult<Self> {
                if !bytes.len().is_multiple_of(dtype.width()) {
                    return Err(LadduDataError::Schema("invalid integer column byte count".into()));
                }
                Ok(match dtype { $(ColumnDType::$variant =>
                    Self::$variant(bytes.as_chunks::<{ std::mem::size_of::<$ty>() }>().0.iter()
                        .map(|bytes| <$ty>::from_le_bytes(*bytes)).collect()),)* })
            }

            /// Copies selected rows in the requested order.
            ///
            /// # Panics
            /// Panics when a requested row is outside the column.
            pub fn select(&self, rows: &[usize]) -> Self {
                match self { $(Self::$variant(values) => Self::$variant(rows.iter().map(|&row| values[row]).collect()),)* }
            }

            /// Copies a half-open range of values.
            ///
            /// # Panics
            /// Panics when the range is invalid.
            pub fn slice(&self, start: usize, end: usize) -> Self {
                match self { $(Self::$variant(values) => Self::$variant(Arc::from(&values[start..end])),)* }
            }

        }

        pub(crate) enum ColumnBuffer {
            $($variant(Vec<$ty>),)*
        }

        impl ColumnBuffer {
            pub(crate) fn new(dtype: ColumnDType, capacity: usize) -> Self {
                match dtype { $(ColumnDType::$variant => Self::$variant(Vec::with_capacity(capacity)),)* }
            }

            pub(crate) fn push(&mut self, value: ColumnValue) -> LadduDataResult<()> {
                match (self, value) {
                    $((Self::$variant(values), ColumnValue::$variant(value)) => values.push(value),)*
                    _ => return Err(LadduDataError::Schema("typed row value dtype does not match schema".into())),
                }
                Ok(())
            }

            pub(crate) fn extend(&mut self, column: &Column) -> LadduDataResult<()> {
                match (self, column) {
                    $((Self::$variant(values), Column::$variant(column)) => values.extend_from_slice(column),)*
                    _ => return Err(LadduDataError::Schema("typed column dtype does not match schema".into())),
                }
                Ok(())
            }

            pub(crate) fn finish(self) -> Column {
                match self { $(Self::$variant(values) => Column::$variant(values.into()),)* }
            }
        }
    };
}

integer_columns!(
    (I8, i8, "int8", "i8"),
    (U8, u8, "uint8", "u8"),
    (I16, i16, "int16", "i16"),
    (U16, u16, "uint16", "u16"),
    (I32, i32, "int32", "i32"),
    (U32, u32, "uint32", "u32"),
    (I64, i64, "int64", "i64"),
    (U64, u64, "uint64", "u64"),
);

impl fmt::Display for ColumnDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Column {
    /// Returns whether the column contains no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a correctly typed empty column.
    pub fn empty(dtype: ColumnDType) -> Self {
        ColumnBuffer::new(dtype, 0).finish()
    }

    /// Concatenates columns of one exact dtype.
    ///
    /// # Errors
    /// Returns an error if any column has a different dtype.
    pub fn concat(dtype: ColumnDType, columns: &[&Self]) -> LadduDataResult<Self> {
        let mut output = ColumnBuffer::new(dtype, columns.iter().map(|column| column.len()).sum());
        for column in columns {
            output.extend(column)?;
        }
        Ok(output.finish())
    }
}
