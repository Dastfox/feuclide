//! Reads the frame drops the recorder wrote (`fk_render::drops`) and tells them as a table:
//! each drop, or, grouped, how often each cause came back.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use fk_render::drops::DropReport;

/// Every drop's report in `dir`, newest first, by its file's name.
///
/// # Errors
///
/// When `dir` cannot be read; a report that cannot be parsed is left out with a warning on
/// standard error.
pub fn read_dir(dir: &Path) -> Result<Vec<(String, DropReport)>, String> {
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    // `<unix seconds>-<n>-<kind>`: the name's numbers sort them.
    paths.sort_by_key(|path| {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let mut numbers = stem.split('-').map(|part| part.parse::<u64>().unwrap_or(0));
        (numbers.next().unwrap_or(0), numbers.next().unwrap_or(0))
    });
    paths.reverse();
    let mut reports = Vec::new();
    for path in paths {
        match DropReport::read(&path) {
            Ok(report) => {
                let name = path.file_stem().unwrap_or_default().to_string_lossy();
                reports.push((name.into_owned(), report));
            }
            Err(error) => eprintln!("{}: {error}", path.display()),
        }
    }
    Ok(reports)
}

/// What was most over usual in `report`, as `what +ms`, or `–`.
pub fn cause(report: &DropReport) -> String {
    report
        .over
        .first()
        .filter(|over| over.grew() > 0.0)
        .map_or("–".to_owned(), |over| {
            format!("{} +{:.1}", over.what, over.grew())
        })
}

/// One drop a row: its file, kind, length against the usual, build, cause, events and where.
pub fn table(reports: &[(String, DropReport)]) -> String {
    let mut table = String::from(
        "| file | kind | ms (usual) | build | most over | events | place | eye |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (name, report) in reports {
        let events: Vec<String> = report
            .events
            .iter()
            .map(|(name, count)| format!("{name} {count}"))
            .collect();
        let note = |key: &str| report.notes.get(key).map_or("–", String::as_str);
        let _ = writeln!(
            table,
            "| {name} | {}{} | {:.0} ({:.1}) | {} | {} | {} | {} | {} |",
            report.kind,
            if report.drops > 1 {
                format!(" ×{}", report.drops)
            } else {
                String::new()
            },
            report.ms,
            report.usual_ms,
            if report.optimized { "release" } else { "debug" },
            cause(report),
            if events.is_empty() {
                "–".to_owned()
            } else {
                events.join(", ")
            },
            note("place"),
            note("eye"),
        );
    }
    table
}

/// How often each cause and each event came with a drop, most often first.
pub fn grouped(reports: &[(String, DropReport)]) -> String {
    let mut causes: BTreeMap<String, usize> = BTreeMap::new();
    let mut events: BTreeMap<String, usize> = BTreeMap::new();
    for (_, report) in reports {
        let what = report
            .over
            .first()
            .filter(|over| over.grew() > 0.0)
            .map_or("–".to_owned(), |over| over.what.clone());
        *causes.entry(format!("{} {what}", report.kind)).or_default() += 1;
        for name in report.events.keys() {
            *events.entry(name.clone()).or_default() += 1;
        }
    }
    let mut text = String::from("| most over | drops |\n|---|---:|\n");
    let mut causes: Vec<_> = causes.into_iter().collect();
    causes.sort_by(|a, b| b.1.cmp(&a.1));
    for (cause, count) in causes {
        let _ = writeln!(text, "| {cause} | {count} |");
    }
    text.push_str("\n| event in the frame | drops |\n|---|---:|\n");
    let mut events: Vec<_> = events.into_iter().collect();
    events.sort_by(|a, b| b.1.cmp(&a.1));
    for (event, count) in events {
        let _ = writeln!(text, "| {event} | {count} |");
    }
    text
}

#[cfg(test)]
mod tests {
    use fk_render::drops::Over;

    use super::*;

    fn spike(what: &str, ms: f64) -> DropReport {
        DropReport {
            kind: "spike".to_owned(),
            ms: 200.0,
            usual_ms: 16.0,
            over: vec![
                Over {
                    what: what.to_owned(),
                    ms,
                    usual_ms: 2.0,
                },
                Over {
                    what: "gpu opaque".to_owned(),
                    ms: 8.0,
                    usual_ms: 8.0,
                },
            ],
            events: [("pipelines built".to_owned(), 3)].into(),
            notes: [("place".to_owned(), "road".to_owned())].into(),
            ..DropReport::default()
        }
    }

    #[test]
    fn a_written_drop_is_read_and_its_cause_named() {
        let dir = std::env::temp_dir().join(format!("fk-drops-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, report) in [
            ("100-1-spike", spike("cpu update", 190.0)),
            ("200-2-spike", spike("cpu render", 50.0)),
        ] {
            let text = toml::to_string(&report).unwrap();
            std::fs::write(dir.join(format!("{name}.toml")), text).unwrap();
        }
        std::fs::write(dir.join("200-2-spike.csv"), "not a report").unwrap();
        let reports = read_dir(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].0, "200-2-spike", "newest first");
        assert_eq!(cause(&reports[1].1), "cpu update +188.0");
        let table = table(&reports);
        assert!(table.contains("cpu update +188.0"));
        assert!(table.contains("pipelines built 3"));
        assert!(table.contains("| road |"));
        let grouped = grouped(&reports);
        assert!(grouped.contains("| spike cpu update | 1 |"));
        assert!(grouped.contains("| pipelines built | 2 |"));
    }
}
