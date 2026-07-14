//! Minimal JSON value type with a parser and two serializers (pretty and
//! compact). Written against untrusted input: byte-position errors, a
//! nesting-depth cap, exact integer preservation up to i128, and full
//! `\uXXXX` escape handling including surrogate pairs. Object key order is
//! preserved (a `Vec`, not a map) so reports render deterministically and
//! safetensors headers round-trip in file order.

use std::fmt;

/// A parsed or programmatically built JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Integers are kept exact (offsets in safetensors headers must not be
    /// rounded through f64).
    Int(i128),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Parse failure with the byte offset where it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError {
    pub msg: String,
    pub offset: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.msg, self.offset)
    }
}

const MAX_DEPTH: usize = 128;

impl Json {
    /// Parse a complete JSON document. Trailing whitespace is allowed,
    /// trailing garbage is not.
    pub fn parse(input: &[u8]) -> Result<Json, JsonError> {
        let mut p = Parser { b: input, pos: 0 };
        p.skip_ws();
        let v = p.value(0)?;
        p.skip_ws();
        if p.pos != p.b.len() {
            return Err(p.err("trailing data after JSON document"));
        }
        Ok(v)
    }

    /// Object member lookup (first match); `None` on non-objects.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i128> {
        match self {
            Json::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// Serialize with 2-space indentation and a trailing newline-free string.
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        write_value(self, Some(0), &mut out);
        out
    }

    /// Serialize on a single line with no optional whitespace.
    pub fn compact(&self) -> String {
        let mut out = String::new();
        write_value(self, None, &mut out);
        out
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Json::Str(s.to_string())
    }
}

impl From<u64> for Json {
    fn from(n: u64) -> Self {
        Json::Int(n as i128)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> JsonError {
        JsonError {
            msg: msg.to_string(),
            offset: self.pos,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.b.get(self.pos) {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_lit(&mut self, lit: &[u8], v: Json) -> Result<Json, JsonError> {
        if self.b[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(v)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        match self.peek() {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.expect_lit(b"true", Json::Bool(true)),
            Some(b'f') => self.expect_lit(b"false", Json::Bool(false)),
            Some(b'n') => self.expect_lit(b"null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.pos += 1; // '{'
        let mut members = Vec::new();
        self.skip_ws();
        if self.eat(b'}') {
            return Ok(Json::Obj(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.string()?;
            self.skip_ws();
            if !self.eat(b':') {
                return Err(self.err("expected ':' after key"));
            }
            self.skip_ws();
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.skip_ws();
            if self.eat(b',') {
                continue;
            }
            if self.eat(b'}') {
                return Ok(Json::Obj(members));
            }
            return Err(self.err("expected ',' or '}' in object"));
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.eat(b']') {
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            if self.eat(b',') {
                continue;
            }
            if self.eat(b']') {
                return Ok(Json::Arr(items));
            }
            return Err(self.err("expected ',' or ']' in array"));
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.pos += 1; // '"'
        let mut s = String::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                // High surrogate: a \uXXXX low surrogate must follow.
                                if !(self.eat(b'\\') && self.eat(b'u')) {
                                    return Err(self.err("lone high surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                char::from_u32(cp).ok_or_else(|| self.err("bad surrogate pair"))?
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("lone low surrogate"));
                            } else {
                                char::from_u32(hi).ok_or_else(|| self.err("bad code point"))?
                            };
                            s.push(ch);
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                0x00..=0x1F => return Err(self.err("raw control character in string")),
                _ => {
                    // Re-decode the UTF-8 sequence starting at the byte we consumed.
                    let start = self.pos - 1;
                    let len = utf8_len(c).ok_or_else(|| self.err("invalid UTF-8"))?;
                    let end = start + len;
                    if end > self.b.len() {
                        return Err(self.err("truncated UTF-8 sequence"));
                    }
                    let chunk = std::str::from_utf8(&self.b[start..end])
                        .map_err(|_| self.err("invalid UTF-8"))?;
                    s.push_str(chunk);
                    self.pos = end;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self
                .peek()
                .ok_or_else(|| self.err("truncated \\u escape"))?;
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| self.err("bad hex digit"))?;
            v = v * 16 + d;
            self.pos += 1;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        self.eat(b'-');
        if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return Err(self.err("expected digit"));
        }
        // No leading zeros: "0" alone or a non-zero first digit.
        if self.peek() == Some(b'0') {
            self.pos += 1;
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("leading zero in number"));
            }
        } else {
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("expected digit after '.'"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("expected exponent digit"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.pos]).unwrap();
        if is_float {
            let f: f64 = text.parse().map_err(|_| self.err("invalid number"))?;
            Ok(Json::Float(f))
        } else {
            match text.parse::<i128>() {
                Ok(n) => Ok(Json::Int(n)),
                // Integers beyond i128 degrade to f64 rather than failing.
                Err(_) => Ok(Json::Float(
                    text.parse::<f64>()
                        .map_err(|_| self.err("invalid number"))?,
                )),
            }
        }
    }
}

fn utf8_len(first: u8) -> Option<usize> {
    match first {
        0x20..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Serializer
// ---------------------------------------------------------------------------

fn write_value(v: &Json, indent: Option<usize>, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(n) => out.push_str(&n.to_string()),
        Json::Float(f) => {
            if f.is_finite() {
                let s = format!("{f}");
                out.push_str(&s);
                // "1" would re-parse as an integer; keep floats floats.
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    out.push_str(".0");
                }
            } else {
                out.push_str("null"); // NaN/Inf are not representable in JSON
            }
        }
        Json::Str(s) => write_string(s, out),
        Json::Arr(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Some(level) = indent {
                    newline_indent(level + 1, out);
                    write_value(item, Some(level + 1), out);
                } else {
                    write_value(item, None, out);
                }
            }
            if let Some(level) = indent {
                newline_indent(level, out);
            }
            out.push(']');
        }
        Json::Obj(members) => {
            if members.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, val)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Some(level) = indent {
                    newline_indent(level + 1, out);
                    write_string(k, out);
                    out.push_str(": ");
                    write_value(val, Some(level + 1), out);
                } else {
                    write_string(k, out);
                    out.push(':');
                    write_value(val, None, out);
                }
            }
            if let Some(level) = indent {
                newline_indent(level, out);
            }
            out.push('}');
        }
    }
}

fn newline_indent(level: usize, out: &mut String) {
    out.push('\n');
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Json {
        Json::parse(s.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn parses_scalars_and_keywords() {
        assert_eq!(parse("null"), Json::Null);
        assert_eq!(parse("true"), Json::Bool(true));
        assert_eq!(parse(" false "), Json::Bool(false));
        assert_eq!(parse("42"), Json::Int(42));
        assert_eq!(parse("-7"), Json::Int(-7));
        assert_eq!(parse("2.5"), Json::Float(2.5));
        assert_eq!(parse("1e3"), Json::Float(1000.0));
    }

    #[test]
    fn preserves_u64_offsets_exactly() {
        // 2^63 + 3 loses precision through f64; safetensors offsets must not.
        let v = parse("9223372036854775811");
        assert_eq!(v, Json::Int(9_223_372_036_854_775_811));
    }

    #[test]
    fn object_key_order_is_preserved_and_accessors_work() {
        let v = parse(r#"{"z":1,"a":"F32","m":[2,3]}"#);
        let Json::Obj(ref members) = v else {
            panic!("not an object")
        };
        let keys: Vec<&str> = members.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["z", "a", "m"]);
        assert_eq!(v.get("a").and_then(Json::as_str), Some("F32"));
        assert_eq!(
            v.get("m").and_then(Json::as_arr).unwrap()[0].as_int(),
            Some(2)
        );
        assert!(v.get("missing").is_none());
    }

    #[test]
    fn decodes_escapes_surrogate_pairs_and_multibyte_utf8() {
        assert_eq!(parse(r#""a\nb\t\"\\""#), Json::Str("a\nb\t\"\\".into()));
        assert_eq!(parse(r#""é""#), Json::Str("é".into()));
        // U+1F600 as a surrogate pair.
        assert_eq!(parse(r#""😀""#), Json::Str("😀".into()));
        assert_eq!(
            parse("\"多言語テンソル\""),
            Json::Str("多言語テンソル".into())
        );
    }

    #[test]
    fn rejects_malformed_documents_with_byte_offsets() {
        // Lone surrogates and bad escapes.
        assert!(Json::parse(br#""\ud800""#).is_err());
        assert!(Json::parse(br#""\udc00""#).is_err());
        assert!(Json::parse(br#""\q""#).is_err());
        // Trailing garbage, leading zeros, raw control characters.
        assert!(Json::parse(b"{} x").is_err());
        assert!(Json::parse(b"01").is_err());
        assert!(Json::parse(b"\"a\x01b\"").is_err());
        // Errors carry the byte offset where parsing failed.
        assert_eq!(Json::parse(br#"{"a": }"#).unwrap_err().offset, 6);
    }

    #[test]
    fn depth_cap_stops_a_nesting_bomb() {
        let bomb = "[".repeat(100_000);
        let err = Json::parse(bomb.as_bytes()).unwrap_err();
        assert!(err.msg.contains("deep"), "got: {err}");
    }

    #[test]
    fn pretty_and_compact_round_trip() {
        let src = r#"{"name":"fc1.weight","shape":[8,16],"ok":true,"lr":0.001,"note":null}"#;
        let v = parse(src);
        assert_eq!(Json::parse(v.pretty().as_bytes()).unwrap(), v);
        assert_eq!(v.compact(), src);
    }

    #[test]
    fn serializer_escapes_and_marks_floats() {
        assert_eq!(Json::Str("a\"b\u{1}".into()).compact(), "\"a\\\"b\\u0001\"");
        // A whole-number float keeps a decimal point so it re-parses as a float.
        assert_eq!(Json::Float(3.0).compact(), "3.0");
        assert_eq!(Json::Float(f64::NAN).compact(), "null");
    }
}
