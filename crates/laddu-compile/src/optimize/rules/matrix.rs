use laddu_expr::{ExprId, ExprMetadata, ExprNode, ExprSourceKind, ValueKind};

use crate::CompileResult;

use super::super::rewrite::alias_or_preserve;
use super::super::{MatrixVectorRule, Rewrite, RewriteContext, RewriteRule};

use super::add_mul::ReplacementFragment;

impl RewriteRule for MatrixVectorRule {
    fn name(&self) -> &'static str {
        "matrix-vector"
    }

    fn rewrite(
        &self,
        node: &ExprNode,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        match node {
            ExprNode::MatMul { lhs, rhs } => self.rewrite_matmul(*lhs, *rhs, metadata, context),
            ExprNode::MatVec { matrix, vector } => {
                self.rewrite_matvec(*matrix, *vector, metadata, context)
            }
            ExprNode::Dot { lhs, rhs } => self.rewrite_dot(*lhs, *rhs, metadata, context),
            ExprNode::Component { input, index } => {
                self.rewrite_component(*input, *index, metadata, context)
            }
            ExprNode::MatrixElement { input, row, col } => {
                self.rewrite_matrix_element(*input, *row, *col, metadata, context)
            }
            _ => Ok(Rewrite::Keep),
        }
    }
}

impl MatrixVectorRule {
    fn rewrite_component(
        &self,
        input: ExprId,
        index: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(ExprNode::MatVec { matrix, vector }) = context.node(input) else {
            return Ok(Rewrite::Keep);
        };
        if !matches!(context.node(*matrix), Some(ExprNode::Matrix { .. }))
            || !matches!(context.node(*vector), Some(ExprNode::Vector { .. }))
        {
            return Ok(Rewrite::Keep);
        }
        let (Some((rows, cols)), Some(len)) = (
            Self::matrix_dims(context, *matrix),
            Self::vector_len(context, *vector),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if index >= rows || cols != len || cols == 0 {
            return Ok(Rewrite::Keep);
        }

        let mut builder = ReplacementFragment::new(context);
        let terms = (0..cols)
            .map(|col| {
                let lhs = Self::matrix_element(*matrix, index, col, &mut builder);
                let rhs = Self::vector_element(*vector, col, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::Component { input, index },
            metadata,
            builder,
            context,
        )
    }

    fn rewrite_matrix_element(
        &self,
        input: ExprId,
        row: usize,
        col: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Some(ExprNode::MatMul { lhs, rhs }) = context.node(input) else {
            return Ok(Rewrite::Keep);
        };
        if !matches!(context.node(*lhs), Some(ExprNode::Matrix { .. }))
            || !matches!(context.node(*rhs), Some(ExprNode::Matrix { .. }))
        {
            return Ok(Rewrite::Keep);
        }
        let (Some((lhs_rows, lhs_cols)), Some((rhs_rows, rhs_cols))) = (
            Self::matrix_dims(context, *lhs),
            Self::matrix_dims(context, *rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if row >= lhs_rows || col >= rhs_cols || lhs_cols != rhs_rows || lhs_cols == 0 {
            return Ok(Rewrite::Keep);
        }

        let mut builder = ReplacementFragment::new(context);
        let terms = (0..lhs_cols)
            .map(|inner| {
                let lhs = Self::matrix_element(*lhs, row, inner, &mut builder);
                let rhs = Self::matrix_element(*rhs, inner, col, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::MatrixElement { input, row, col },
            metadata,
            builder,
            context,
        )
    }

    fn rewrite_matmul(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if Self::matrix_is_identity(context, lhs) && Self::matrix_dims(context, rhs).is_some() {
            return Ok(alias_or_preserve(rhs, metadata, context));
        }
        if Self::matrix_is_identity(context, rhs) && Self::matrix_dims(context, lhs).is_some() {
            return Ok(alias_or_preserve(lhs, metadata, context));
        }

        let (Some((lhs_rows, lhs_cols)), Some((rhs_rows, rhs_cols))) = (
            Self::matrix_dims(context, lhs),
            Self::matrix_dims(context, rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if lhs_cols != rhs_rows {
            return Ok(Rewrite::Keep);
        }
        if Self::matrix_is_zero(context, lhs) || Self::matrix_is_zero(context, rhs) {
            return Ok(Self::zero_matrix_rewrite(
                lhs_rows, rhs_cols, metadata, context,
            ));
        }
        Ok(Rewrite::Keep)
    }

    fn rewrite_matvec(
        &self,
        matrix: ExprId,
        vector: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        if Self::matrix_is_identity(context, matrix) && Self::vector_len(context, vector).is_some()
        {
            return Ok(alias_or_preserve(vector, metadata, context));
        }

        let (Some((rows, cols)), Some(len)) = (
            Self::matrix_dims(context, matrix),
            Self::vector_len(context, vector),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if cols != len {
            return Ok(Rewrite::Keep);
        }
        if Self::matrix_is_zero(context, matrix) || Self::vector_is_zero(context, vector) {
            return Ok(Self::zero_vector_rewrite(rows, metadata, context));
        }
        self.expand_matvec(matrix, vector, rows, cols, metadata, context)
    }

    fn rewrite_dot(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let (Some(lhs_len), Some(rhs_len)) = (
            Self::vector_len(context, lhs),
            Self::vector_len(context, rhs),
        ) else {
            return Ok(Rewrite::Keep);
        };
        if lhs_len != rhs_len {
            return Ok(Rewrite::Keep);
        }
        if Self::vector_is_zero(context, lhs) || Self::vector_is_zero(context, rhs) {
            return Ok(Rewrite::Replace {
                node: ExprNode::RealConst(0.0),
                metadata: metadata.clone(),
            });
        }
        self.expand_dot(lhs, rhs, lhs_len, metadata, context)
    }

    fn expand_dot(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        len: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let terms = (0..len)
            .map(|index| {
                let lhs = Self::vector_element(lhs, index, &mut builder);
                let rhs = Self::vector_element(rhs, index, &mut builder);
                builder.push(
                    ExprNode::NaryMul {
                        factors: vec![lhs, rhs],
                    },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::NaryAdd { terms }, metadata.clone());
        self.cost_gated_fragment(ExprNode::Dot { lhs, rhs }, metadata, builder, context)
    }

    fn expand_matvec(
        &self,
        matrix: ExprId,
        vector: ExprId,
        rows: usize,
        cols: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let mut builder = ReplacementFragment::new(context);
        let elements = (0..rows)
            .map(|row| {
                let terms = (0..cols)
                    .map(|col| {
                        let lhs = Self::matrix_element(matrix, row, col, &mut builder);
                        let rhs = Self::vector_element(vector, col, &mut builder);
                        builder.push(
                            ExprNode::NaryMul {
                                factors: vec![lhs, rhs],
                            },
                            ExprMetadata::new(ExprSourceKind::Binary),
                        )
                    })
                    .collect();
                builder.push(
                    ExprNode::NaryAdd { terms },
                    ExprMetadata::new(ExprSourceKind::Binary),
                )
            })
            .collect();
        builder.push(ExprNode::Vector { elements }, metadata.clone());
        self.cost_gated_fragment(
            ExprNode::MatVec { matrix, vector },
            metadata,
            builder,
            context,
        )
    }

    fn cost_gated_fragment(
        &self,
        original: ExprNode,
        metadata: &ExprMetadata,
        builder: ReplacementFragment<'_>,
        context: &RewriteContext<'_>,
    ) -> CompileResult<Rewrite> {
        let Rewrite::ReplaceMany { nodes } = builder.into_rewrite() else {
            unreachable!("replacement builder always produces fragments")
        };
        let original_cost = context.local_node_cost(original, metadata.clone())?;
        let candidate_cost = context.local_fragment_cost(&nodes)?;
        if candidate_cost.is_better_than(&original_cost) {
            Ok(Rewrite::ReplaceMany { nodes })
        } else {
            Ok(Rewrite::Keep)
        }
    }

    fn zero_vector_rewrite(
        len: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let zero = builder.push(
            ExprNode::RealConst(0.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        builder.push(
            ExprNode::Vector {
                elements: vec![zero; len],
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn zero_matrix_rewrite(
        rows: usize,
        cols: usize,
        metadata: &ExprMetadata,
        context: &RewriteContext<'_>,
    ) -> Rewrite {
        let mut builder = ReplacementFragment::new(context);
        let zero = builder.push(
            ExprNode::RealConst(0.0),
            ExprMetadata::new(ExprSourceKind::Const),
        );
        builder.push(
            ExprNode::Matrix {
                rows,
                cols,
                elements: vec![zero; rows * cols],
            },
            metadata.clone(),
        );
        builder.into_rewrite()
    }

    fn vector_len(context: &RewriteContext<'_>, id: ExprId) -> Option<usize> {
        match context.facts(id)?.value_kind {
            ValueKind::Vector { len } => Some(len),
            _ => None,
        }
    }

    fn matrix_dims(context: &RewriteContext<'_>, id: ExprId) -> Option<(usize, usize)> {
        match context.facts(id)?.value_kind {
            ValueKind::Matrix { rows, cols } => Some((rows, cols)),
            _ => None,
        }
    }

    fn vector_element(id: ExprId, index: usize, builder: &mut ReplacementFragment<'_>) -> ExprId {
        if let Some(ExprNode::Vector { elements }) = builder.context.node(id)
            && let Some(element) = elements.get(index).copied()
        {
            return element;
        }
        builder.push(
            ExprNode::Component { input: id, index },
            ExprMetadata::new(ExprSourceKind::Vector),
        )
    }

    fn matrix_element(
        id: ExprId,
        row: usize,
        col: usize,
        builder: &mut ReplacementFragment<'_>,
    ) -> ExprId {
        if let Some(ExprNode::Matrix { cols, elements, .. }) = builder.context.node(id)
            && let Some(element) = row
                .checked_mul(*cols)
                .and_then(|base| base.checked_add(col))
                .and_then(|index| elements.get(index))
                .copied()
        {
            return element;
        }
        builder.push(
            ExprNode::MatrixElement {
                input: id,
                row,
                col,
            },
            ExprMetadata::new(ExprSourceKind::Matrix),
        )
    }

    fn vector_is_zero(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Vector { elements }) = context.node(id) else {
            return false;
        };
        elements
            .iter()
            .all(|element| context.node(*element).is_some_and(ExprNode::is_zero))
    }

    fn matrix_is_zero(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Matrix { elements, .. }) = context.node(id) else {
            return false;
        };
        elements
            .iter()
            .all(|element| context.node(*element).is_some_and(ExprNode::is_zero))
    }

    fn matrix_is_identity(context: &RewriteContext<'_>, id: ExprId) -> bool {
        let Some(ExprNode::Matrix {
            rows,
            cols,
            elements,
        }) = context.node(id)
        else {
            return false;
        };
        if rows != cols || elements.len() != rows * cols {
            return false;
        }
        for row in 0..*rows {
            for col in 0..*cols {
                let Some(node) = context.node(elements[row * cols + col]) else {
                    return false;
                };
                if row == col {
                    if !ExprNode::is_one(node) {
                        return false;
                    }
                } else if !ExprNode::is_zero(node) {
                    return false;
                }
            }
        }
        true
    }
}
