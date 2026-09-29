//! Compile-time checked equation syntax for `laddu`'s internal CAS.

use std::collections::BTreeSet;

use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Group, Span, TokenStream as Tokens, TokenTree};
use quote::quote;
use syn::{Expr, ExprBinary, ExprCall, ExprPath, Lit, spanned::Spanned};

/// Declares a family of checked, optionally bidirectional equations.
///
/// Identifiers on the left are captures. Identifiers on the right and in
/// guards must have been captured on the left. Scalar `+` and `*` are matched
/// modulo associativity and commutativity by the CAS engine.
#[proc_macro]
pub fn cas_rules(input: TokenStream) -> TokenStream {
    match compile_rules(Tokens::from(input)) {
        Ok(expanded) => expanded.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn compile_rules(input: Tokens) -> syn::Result<Tokens> {
    let mut result = Vec::new();
    let mut statement = Vec::new();
    for token in input {
        if matches!(&token, TokenTree::Punct(punct) if punct.as_char() == ';') {
            if !statement.is_empty() {
                result.push(compile_rule(std::mem::take(&mut statement))?);
            }
        } else {
            statement.push(token);
        }
    }
    if !statement.is_empty() {
        result.push(compile_rule(statement)?);
    }
    Ok(quote! { vec![#(#result),*] })
}

fn compile_rule(tokens: Vec<TokenTree>) -> syn::Result<Tokens> {
    let arrow = tokens.windows(3).position(|window| {
        punct(&window[0], '<') && punct(&window[1], '=') && punct(&window[2], '>')
    });
    let (left, right, bidirectional) = if let Some(index) = arrow {
        (&tokens[..index], &tokens[index + 3..], true)
    } else if let Some(index) = tokens
        .windows(2)
        .position(|window| punct(&window[0], '=') && punct(&window[1], '>'))
    {
        (&tokens[..index], &tokens[index + 2..], false)
    } else {
        return Err(syn::Error::new(
            tokens.first().map_or(Span::call_site(), TokenTree::span),
            "CAS rule needs `=>` or `<=>`",
        ));
    };
    let guard_at = right
        .iter()
        .position(|token| matches!(token, TokenTree::Ident(ident) if ident == "if"));
    let (right, guard) = if let Some(index) = guard_at {
        (&right[..index], Some(&right[index + 1..]))
    } else {
        (right, None)
    };
    let lhs: Expr = syn::parse2(normalize_powers(left.iter().cloned().collect())?)?;
    let rhs: Expr = syn::parse2(normalize_powers(right.iter().cloned().collect())?)?;
    let mut bound = BTreeSet::new();
    collect_captures(&lhs, &mut bound)?;
    check_captures(&rhs, &bound)?;
    let guard_tokens = if let Some(guard) = guard {
        let expression: Expr = syn::parse2(guard.iter().cloned().collect())?;
        emit_guard(&expression, &bound)?
    } else {
        quote!(crate::cas::rules::Guard::Always)
    };
    let lhs_tokens = emit_pattern(&lhs)?;
    let rhs_tokens = emit_pattern(&rhs)?;
    let name = left.iter().cloned().collect::<Tokens>().to_string();
    Ok(quote! {
        crate::cas::rules::Rule::new(
            #name,
            #lhs_tokens,
            #rhs_tokens,
            #bidirectional,
            #guard_tokens,
        )
    })
}

// Rust gives `^` bitwise precedence. In the CAS DSL it denotes exponentiation
// and binds more tightly than multiplication and addition.
fn normalize_powers(tokens: Tokens) -> syn::Result<Tokens> {
    let input = tokens
        .into_iter()
        .map(|token| match token {
            TokenTree::Group(group) => {
                let mut rewritten =
                    Group::new(group.delimiter(), normalize_powers(group.stream())?);
                rewritten.set_span(group.span());
                Ok(TokenTree::Group(rewritten))
            }
            other => Ok(other),
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let mut out = Vec::new();
    let mut index = 0;
    while index < input.len() {
        if punct(&input[index], '^') {
            let left = take_left_atom(&mut out)?;
            let caret = input[index].clone();
            index += 1;
            let right = take_right_atom(&input, &mut index)?;
            if input.get(index).is_some_and(|token| punct(token, '^')) {
                return Err(syn::Error::new(
                    caret.span(),
                    "parenthesize chained CAS powers",
                ));
            }
            let mut expression = Tokens::new();
            expression.extend(left);
            expression.extend(std::iter::once(caret));
            expression.extend(right);
            out.push(TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                expression,
            )));
        } else {
            out.push(input[index].clone());
            index += 1;
        }
    }
    Ok(out.into_iter().collect())
}

fn take_left_atom(out: &mut Vec<TokenTree>) -> syn::Result<Vec<TokenTree>> {
    let last = out
        .pop()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "missing CAS power base"))?;
    if matches!(&last, TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis)
        && matches!(out.last(), Some(TokenTree::Ident(_)))
    {
        Ok(vec![out.pop().expect("function name exists"), last])
    } else {
        Ok(vec![last])
    }
}

fn take_right_atom(input: &[TokenTree], index: &mut usize) -> syn::Result<Vec<TokenTree>> {
    let token = input
        .get(*index)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "missing CAS exponent"))?
        .clone();
    *index += 1;
    if punct(&token, '-') {
        let next = input
            .get(*index)
            .ok_or_else(|| syn::Error::new(token.span(), "missing CAS exponent"))?
            .clone();
        *index += 1;
        return Ok(vec![token, next]);
    }
    if matches!(token, TokenTree::Ident(_))
        && matches!(input.get(*index), Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Parenthesis)
    {
        let group = input[*index].clone();
        *index += 1;
        Ok(vec![token, group])
    } else {
        Ok(vec![token])
    }
}

fn punct(token: &TokenTree, expected: char) -> bool {
    matches!(token, TokenTree::Punct(punct) if punct.as_char() == expected)
}

fn collect_captures(expr: &Expr, found: &mut BTreeSet<String>) -> syn::Result<()> {
    match expr {
        Expr::Path(path) => {
            let name = simple_name(path)?;
            if name != "I" {
                found.insert(name);
            }
        }
        Expr::Binary(binary) => {
            collect_captures(&binary.left, found)?;
            collect_captures(&binary.right, found)?;
        }
        Expr::Unary(unary) => collect_captures(&unary.expr, found)?,
        Expr::Call(call) => {
            function_name(call)?;
            for argument in &call.args {
                collect_captures(argument, found)?;
            }
        }
        Expr::Paren(paren) => collect_captures(&paren.expr, found)?,
        Expr::Group(group) => collect_captures(&group.expr, found)?,
        Expr::Lit(_) => (),
        _ => return Err(syn::Error::new(expr.span(), "unsupported CAS expression")),
    }
    Ok(())
}

fn check_captures(expr: &Expr, bound: &BTreeSet<String>) -> syn::Result<()> {
    let mut used = BTreeSet::new();
    collect_captures(expr, &mut used)?;
    if let Some(missing) = used.difference(bound).next() {
        return Err(syn::Error::new(
            expr.span(),
            format!("unbound CAS capture `{missing}`"),
        ));
    }
    Ok(())
}

fn simple_name(path: &ExprPath) -> syn::Result<String> {
    if path.path.segments.len() != 1 {
        return Err(syn::Error::new(path.span(), "expected one CAS symbol"));
    }
    Ok(path.path.segments[0].ident.to_string())
}

fn function_name(call: &ExprCall) -> syn::Result<String> {
    let Expr::Path(path) = &*call.func else {
        return Err(syn::Error::new(
            call.func.span(),
            "expected CAS function name",
        ));
    };
    simple_name(path)
}

fn emit_pattern(expr: &Expr) -> syn::Result<Tokens> {
    match expr {
        Expr::Paren(paren) => emit_pattern(&paren.expr),
        Expr::Group(group) => emit_pattern(&group.expr),
        Expr::Path(path) => {
            let name = simple_name(path)?;
            if name == "I" {
                Ok(quote!(crate::cas::rules::Pattern::ImaginaryUnit))
            } else {
                Ok(quote!(crate::cas::rules::Pattern::Capture(#name)))
            }
        }
        Expr::Lit(literal) => {
            let number = match &literal.lit {
                Lit::Float(value) => value.base10_parse::<f64>()?,
                Lit::Int(value) => value.base10_parse::<f64>()?,
                _ => return Err(syn::Error::new(expr.span(), "expected numeric CAS literal")),
            };
            Ok(quote!(crate::cas::rules::Pattern::Real(#number)))
        }
        Expr::Unary(unary) => {
            let operand = emit_pattern(&unary.expr)?;
            match unary.op {
                syn::UnOp::Neg(_) => Ok(quote!(crate::cas::rules::Pattern::Unary(
                    ::laddu_expr::UnaryOp::Neg, Box::new(#operand)
                ))),
                _ => Err(syn::Error::new(
                    expr.span(),
                    "unsupported CAS unary operator",
                )),
            }
        }
        Expr::Binary(binary) => emit_binary(binary),
        Expr::Call(call) => {
            let name = function_name(call)?;
            let args = call
                .args
                .iter()
                .map(emit_pattern)
                .collect::<syn::Result<Vec<_>>>()?;
            let unary = match name.as_str() {
                "sin" => Some(quote!(::laddu_expr::UnaryOp::Sin)),
                "cos" => Some(quote!(::laddu_expr::UnaryOp::Cos)),
                "exp" => Some(quote!(::laddu_expr::UnaryOp::Exp)),
                "log" => Some(quote!(::laddu_expr::UnaryOp::Log)),
                "sqrt" => Some(quote!(::laddu_expr::UnaryOp::Sqrt)),
                "conj" => Some(quote!(::laddu_expr::UnaryOp::Conj)),
                "real" => Some(quote!(::laddu_expr::UnaryOp::Real)),
                "imag" => Some(quote!(::laddu_expr::UnaryOp::Imag)),
                "norm_sqr" => Some(quote!(::laddu_expr::UnaryOp::NormSqr)),
                _ => None,
            };
            if let Some(op) = unary {
                if args.len() != 1 {
                    return Err(syn::Error::new(
                        expr.span(),
                        "CAS unary function needs one argument",
                    ));
                }
                let input = &args[0];
                return Ok(quote!(crate::cas::rules::Pattern::Unary(#op, Box::new(#input))));
            }
            let call2 = match name.as_str() {
                "complex" => quote!(crate::cas::rules::Call2::Complex),
                "matmul" => quote!(crate::cas::rules::Call2::MatMul),
                "matvec" => quote!(crate::cas::rules::Call2::MatVec),
                "dot" => quote!(crate::cas::rules::Call2::Dot),
                "solve" => quote!(crate::cas::rules::Call2::Solve),
                _ => {
                    return Err(syn::Error::new(
                        expr.span(),
                        format!("unknown CAS function `{name}`"),
                    ));
                }
            };
            if args.len() != 2 {
                return Err(syn::Error::new(
                    expr.span(),
                    "CAS binary function needs two arguments",
                ));
            }
            let lhs = &args[0];
            let rhs = &args[1];
            Ok(quote!(crate::cas::rules::Pattern::Call2(#call2, Box::new(#lhs), Box::new(#rhs))))
        }
        _ => Err(syn::Error::new(expr.span(), "unsupported CAS expression")),
    }
}

fn emit_binary(expr: &ExprBinary) -> syn::Result<Tokens> {
    use syn::BinOp;
    let lhs = emit_pattern(&expr.left)?;
    let rhs = emit_pattern(&expr.right)?;
    match &expr.op {
        BinOp::Add(_) => Ok(quote!(crate::cas::rules::Pattern::Sum(vec![#lhs, #rhs]))),
        BinOp::Mul(_) => Ok(quote!(crate::cas::rules::Pattern::Product(
            vec![#lhs, #rhs]
        ))),
        BinOp::Sub(_) => Ok(quote!(crate::cas::rules::Pattern::Binary(
            ::laddu_expr::BinaryOp::Sub, Box::new(#lhs), Box::new(#rhs)
        ))),
        BinOp::Div(_) => Ok(quote!(crate::cas::rules::Pattern::Binary(
            ::laddu_expr::BinaryOp::Div, Box::new(#lhs), Box::new(#rhs)
        ))),
        BinOp::BitXor(_) => {
            let power = match &*expr.right {
                Expr::Lit(literal) => match &literal.lit {
                    Lit::Int(power) => power.base10_parse::<i32>()?,
                    _ => {
                        return Err(syn::Error::new(
                            expr.right.span(),
                            "CAS power needs an integer literal",
                        ));
                    }
                },
                Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
                    let Expr::Lit(literal) = &*unary.expr else {
                        return Err(syn::Error::new(
                            expr.right.span(),
                            "CAS power needs an integer literal",
                        ));
                    };
                    let Lit::Int(power) = &literal.lit else {
                        return Err(syn::Error::new(
                            expr.right.span(),
                            "CAS power needs an integer literal",
                        ));
                    };
                    -power.base10_parse::<i32>()?
                }
                _ => {
                    return Err(syn::Error::new(
                        expr.right.span(),
                        "CAS power needs an integer literal",
                    ));
                }
            };
            Ok(quote!(crate::cas::rules::Pattern::Unary(
                ::laddu_expr::UnaryOp::PowI(#power), Box::new(#lhs)
            )))
        }
        _ => Err(syn::Error::new(
            expr.op.span(),
            "unsupported CAS binary operator",
        )),
    }
}

fn emit_guard(expr: &Expr, bound: &BTreeSet<String>) -> syn::Result<Tokens> {
    match expr {
        Expr::Paren(paren) => emit_guard(&paren.expr, bound),
        Expr::Group(group) => emit_guard(&group.expr, bound),
        Expr::Binary(binary) => {
            let lhs = emit_guard(&binary.left, bound)?;
            let rhs = emit_guard(&binary.right, bound)?;
            match binary.op {
                syn::BinOp::And(_) => {
                    Ok(quote!(crate::cas::rules::Guard::And(Box::new(#lhs), Box::new(#rhs))))
                }
                syn::BinOp::Or(_) => {
                    Ok(quote!(crate::cas::rules::Guard::Or(Box::new(#lhs), Box::new(#rhs))))
                }
                _ => Err(syn::Error::new(
                    expr.span(),
                    "CAS guard supports `&&` and `||`",
                )),
            }
        }
        Expr::Call(call) => {
            let name = function_name(call)?;
            if call.args.len() != 1 {
                return Err(syn::Error::new(expr.span(), "CAS guard needs one capture"));
            }
            let Expr::Path(path) = &call.args[0] else {
                return Err(syn::Error::new(expr.span(), "CAS guard needs one capture"));
            };
            let capture = simple_name(path)?;
            if !bound.contains(&capture) {
                return Err(syn::Error::new(
                    expr.span(),
                    format!("unbound CAS capture `{capture}`"),
                ));
            }
            let guard = match name.as_str() {
                "real" => quote!(crate::cas::rules::Guard::Real(#capture)),
                "nonzero" => quote!(crate::cas::rules::Guard::Nonzero(#capture)),
                "scalar" => quote!(crate::cas::rules::Guard::Scalar(#capture)),
                "zero" => quote!(crate::cas::rules::Guard::Zero(#capture)),
                "identity" => quote!(crate::cas::rules::Guard::Identity(#capture)),
                _ => {
                    return Err(syn::Error::new(
                        expr.span(),
                        format!("unknown CAS guard `{name}`"),
                    ));
                }
            };
            Ok(guard)
        }
        Expr::Lit(literal) if matches!(&literal.lit, Lit::Bool(value) if value.value) => {
            Ok(quote!(crate::cas::rules::Guard::Always))
        }
        _ => Err(syn::Error::new(expr.span(), "unsupported CAS guard")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn rejects_unbound_rhs_and_guard_captures() {
        assert!(
            compile_rules(quote!(x + 0 => y;))
                .unwrap_err()
                .to_string()
                .contains("unbound CAS capture `y`")
        );
        assert!(
            compile_rules(quote!(x => x if real(y);))
                .unwrap_err()
                .to_string()
                .contains("unbound CAS capture `y`")
        );
    }

    #[test]
    fn rejects_unsupported_functions_and_guards() {
        assert!(
            compile_rules(quote!(tan(x) => x;))
                .unwrap_err()
                .to_string()
                .contains("unknown CAS function")
        );
        assert!(
            compile_rules(quote!(x => x if positive(x);))
                .unwrap_err()
                .to_string()
                .contains("unknown CAS guard")
        );
    }

    #[test]
    fn accepts_bidirectional_ac_rules_and_negative_powers() {
        compile_rules(quote!(x * a + x * b <=> x * (a + b); x ^ -2 => 1 / (x ^ 2) if nonzero(x);))
            .unwrap();
    }

    #[test]
    fn rejects_ambiguous_power_towers() {
        assert!(
            compile_rules(quote!(x ^ 2 ^ 3 => x;))
                .unwrap_err()
                .to_string()
                .contains("parenthesize chained CAS powers")
        );
    }
}
