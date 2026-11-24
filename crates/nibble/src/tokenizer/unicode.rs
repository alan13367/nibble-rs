//! Unicode general-category lookups for the pre-tokenizers.

#[rustfmt::skip]
#[path = "unicode_tables.rs"]
mod tables;

fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let cp = u32::from(c);
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{L}`: a letter.
pub(super) fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_table(tables::LETTER, c)
}

/// `\p{M}`: a combining mark.
pub(super) fn is_mark(c: char) -> bool {
    !c.is_ascii() && in_table(tables::MARK, c)
}

/// `\p{N}`: a number.
pub(super) fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(tables::NUMBER, c)
}

/// `\s`: Unicode `White_Space`, which is exactly `char::is_whitespace`.
pub(super) fn is_space(c: char) -> bool {
    c.is_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories() {
        for c in ['a', 'Z', 'é', 'ß', 'Ж', '中', 'ア', 'ㄱ'] {
            assert!(is_letter(c), "{c:?}");
            assert!(!is_number(c), "{c:?}");
        }
        for c in ['0', '9', '٣', '½', 'Ⅻ', '²'] {
            assert!(is_number(c), "{c:?}");
            assert!(!is_letter(c), "{c:?}");
        }
        for c in ['\u{0301}', '\u{093F}', '\u{20DD}'] {
            assert!(is_mark(c), "{c:?}");
            assert!(!is_letter(c), "{c:?}");
        }
        for c in [' ', '!', '_', '€', '😀', '\n'] {
            assert!(!is_letter(c) && !is_number(c) && !is_mark(c), "{c:?}");
        }
        for c in [' ', '\t', '\n', '\r', '\u{00A0}', '\u{3000}', '\u{2028}'] {
            assert!(is_space(c), "{c:?}");
        }
    }
}
