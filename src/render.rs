//! Human-readable rendering for `tensorpeek ls`: an aligned tensor table
//! with a one-line summary, plus the byte/count humanizers it uses.

use crate::report::{filter_match, Report};

/// Render the tensor table for one report.
pub fn tensor_table(report: &Report, filter: Option<&str>) -> String {
    let mut out = String::new();
    let count = report.tensors.len();
    let params = report.parameters();
    out.push_str(&format!(
        "{} · {} · {} tensor{} · {} param{} · {} data\n",
        report.file,
        report.format,
        count,
        if count == 1 { "" } else { "s" },
        human_count(params),
        if params == 1 { "" } else { "s" },
        human_bytes(report.data_bytes),
    ));

    let rows: Vec<[String; 4]> = report
        .tensors
        .iter()
        .filter(|t| filter.map_or(true, |f| filter_match(f, &t.name)))
        .map(|t| {
            [
                if t.name.is_empty() {
                    "-".to_string()
                } else {
                    t.name.clone()
                },
                t.dtype.clone(),
                shape_text(&t.shape),
                human_bytes(t.bytes),
            ]
        })
        .collect();

    if !rows.is_empty() {
        let header = ["NAME", "DTYPE", "SHAPE", "BYTES"];
        let mut widths: Vec<usize> = header.iter().map(|h| h.len()).collect();
        for row in &rows {
            for (w, cell) in widths.iter_mut().zip(row.iter()) {
                *w = (*w).max(cell.chars().count());
            }
        }
        push_row(&mut out, &header.map(String::from), &widths);
        for row in &rows {
            push_row(&mut out, row, &widths);
        }
    }
    for p in &report.problems {
        out.push_str(&format!("problem: {p}\n"));
    }
    out
}

fn push_row(out: &mut String, cells: &[String; 4], widths: &[usize]) {
    for (i, cell) in cells.iter().enumerate() {
        if i + 1 == cells.len() {
            out.push_str(cell);
        } else {
            out.push_str(&format!("{cell:<width$}  ", width = widths[i]));
        }
    }
    out.push('\n');
}

/// `2×3×4`, or `scalar` for a zero-dimensional tensor.
pub fn shape_text(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("×")
}

/// IEC-style byte counts: exact below 1 KiB, one decimal above.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["KiB", "MiB", "GiB", "TiB", "PiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut v = n as f64 / 1024.0;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    format!("{v:.1} {}", UNITS[unit])
}

/// Parameter counts in ML vernacular: 124.4 M, 7.2 B.
pub fn human_count(n: u128) -> String {
    const STEPS: [(u128, &str); 4] = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ];
    for (step, suffix) in STEPS {
        if n >= step {
            return format!("{:.1} {suffix}", n as f64 / step as f64);
        }
    }
    n.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::Tensor;

    #[test]
    fn columns_line_up_across_rows() {
        let mut r = Report::new("m.safetensors", "safetensors", 0);
        r.tensors
            .push(Tensor::new("embed.weight", "f32", vec![32, 8], 0, 1024));
        r.tensors.push(Tensor::new("b", "bf16", vec![], 1024, 2));
        let table = tensor_table(&r, None);
        let lines: Vec<&str> = table.lines().collect();
        let col = lines[1].find("DTYPE").unwrap();
        assert_eq!(lines[2].find("f32"), Some(col));
        assert_eq!(lines[3].find("bf16"), Some(col));
        assert!(lines[3].starts_with("b "), "short names are padded");
        assert!(lines[3].contains("scalar"));
    }

    #[test]
    fn unnamed_npy_tensor_renders_a_dash_and_problems_are_appended() {
        let mut r = Report::new("a.npy", "npy", 0);
        r.tensors.push(Tensor::new("", "f32", vec![3], 0, 12));
        r.problems.push("something is off".into());
        let table = tensor_table(&r, None);
        assert!(
            table.contains("1 tensor ·"),
            "singular, not '1 tensors': {table}"
        );
        assert!(
            table
                .lines()
                .any(|l| l.starts_with("-  ") || l.starts_with("-    ")),
            "table:\n{table}"
        );
        assert!(table.contains("problem: something is off"));
    }

    #[test]
    fn humanizers_cover_the_boundaries() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(17_408), "17.0 KiB");
        assert_eq!(human_bytes(5 * 1024 * 1024 * 1024), "5.0 GiB");
        assert_eq!(human_count(999), "999");
        assert_eq!(human_count(124_400_000), "124.4 M");
        assert_eq!(human_count(7_240_000_000), "7.2 B");
    }
}
