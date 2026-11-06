//! The tuning pane: a folder of TOML tuning files (`fk_app::tuning`), every value a field,
//! edited in place with the files' comments and layout kept (`toml_edit`). Each value's comment
//! is its tooltip; a colour written `0xRRGGBB` gets a colour picker. Saved by hand, or as each
//! edit ends with "Live": an app run with `FK_TUNING_DIR=<that folder>` reloads a file whenever
//! it is saved, so it changes as the values are dragged here. After `bevy_editor_settings`,
//! which edits the editor's own TOML preferences the same way.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use toml_edit::{DocumentMut, Item, Value};

/// The pane's state.
pub struct Tuning {
    dir: PathBuf,
    files: Vec<PathBuf>,
    open: Option<OpenFile>,
    filter: String,
    /// Writes the file whenever an edit ends.
    pub live: bool,
    message: String,
}

struct OpenFile {
    path: PathBuf,
    doc: DocumentMut,
    saved: String,
    modified: Option<SystemTime>,
}

impl OpenFile {
    fn dirty(&self) -> bool {
        self.doc.to_string() != self.saved
    }
}

impl Tuning {
    /// The tuning files in `dir`.
    pub fn new(dir: PathBuf) -> Self {
        let mut tuning = Self {
            dir,
            files: Vec::new(),
            open: None,
            filter: String::new(),
            live: false,
            message: String::new(),
        };
        tuning.rescan();
        tuning
    }

    fn rescan(&mut self) {
        self.files = std::fs::read_dir(&self.dir)
            .map(|entries| {
                let mut files: Vec<_> = entries
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "toml"))
                    .collect();
                files.sort();
                files
            })
            .unwrap_or_default();
    }

    fn open(&mut self, path: &Path) {
        match std::fs::read_to_string(path).map(|text| (text.parse::<DocumentMut>(), text)) {
            Ok((Ok(doc), text)) => {
                self.open = Some(OpenFile {
                    path: path.to_owned(),
                    doc,
                    saved: text,
                    modified: modified(path),
                });
                self.message.clear();
            }
            Ok((Err(e), _)) => self.message = format!("{}: {e}", path.display()),
            Err(e) => self.message = format!("{}: {e}", path.display()),
        }
    }

    /// Opens the file named `name` in the folder.
    pub fn open_named(&mut self, name: &str) {
        let path = self.dir.join(name);
        self.open(&path);
    }

    fn save(&mut self) {
        let Some(open) = &mut self.open else { return };
        let text = open.doc.to_string();
        match std::fs::write(&open.path, &text) {
            Ok(()) => {
                open.saved = text;
                open.modified = modified(&open.path);
                self.message = format!("saved {}", name(&open.path));
            }
            Err(e) => self.message = format!("{}: {e}", open.path.display()),
        }
    }

    /// Draws the pane.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // Changed on disk by someone else (another editor, a script) and not here: read again.
        if let Some(open) = &self.open
            && !open.dirty()
            && modified(&open.path) != open.modified
        {
            let path = open.path.clone();
            self.open(&path);
        }
        ui.horizontal(|ui| {
            ui.heading("Tuning");
            ui.weak(self.dir.display().to_string());
        });
        ui.separator();
        egui::Panel::left("tuning files")
            .resizable(true)
            .default_size(170.0)
            .show(ui, |ui| {
                ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("filter"));
                if ui.small_button("rescan").clicked() {
                    self.rescan();
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut chosen = None;
                    for path in &self.files {
                        let n = name(path);
                        if !self.filter.is_empty() && !n.contains(&self.filter) {
                            continue;
                        }
                        let open = self.open.as_ref().is_some_and(|o| &o.path == path);
                        if ui.selectable_label(open, n).clicked() {
                            chosen = Some(path.clone());
                        }
                    }
                    if let Some(path) = chosen {
                        self.open(&path);
                    }
                });
            });
        let gesture = ui.input(|i| i.pointer.any_down()) || ui.ctx().egui_wants_keyboard_input();
        egui::CentralPanel::default().show(ui, |ui| {
            let Some(open) = &mut self.open else {
                ui.weak("Pick a file. An app run with FK_TUNING_DIR set to this folder reloads each file as it is saved.");
                if !self.message.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, &self.message);
                }
                return;
            };
            let dirty = open.dirty();
            let mut save = false;
            ui.horizontal(|ui| {
                ui.strong(name(&open.path));
                if dirty {
                    ui.colored_label(egui::Color32::from_rgb(240, 180, 80), "●");
                }
                save = ui.add_enabled(dirty, egui::Button::new("Save")).clicked();
                if ui.add_enabled(dirty, egui::Button::new("Revert")).clicked()
                    && let Ok(doc) = open.saved.parse() {
                        open.doc = doc;
                    }
                ui.checkbox(&mut self.live, "Live")
                    .on_hover_text("Save as each edit ends, for an app running with FK_TUNING_DIR.");
                ui.weak(&self.message);
            });
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                table(ui, open.doc.as_table_mut(), "");
            });
            if save || (self.live && dirty && !gesture) {
                self.save();
            }
        });
    }
}

fn name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// The comment lines in a decor's prefix, without their `#`.
fn comment(prefix: Option<&str>) -> Option<String> {
    let lines: Vec<_> = prefix?
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('#'))
        .map(|l| l.trim_start_matches('#').trim())
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn table(ui: &mut egui::Ui, tab: &mut dyn toml_edit::TableLike, path: &str) {
    // Values first, as a grid; then sub-tables, folded.
    let mut subtables = Vec::new();
    egui::Grid::new(("tuning", path))
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (key, item) in tab.iter_mut() {
                let tip = comment(key.leaf_decor().prefix().and_then(|p| p.as_str()));
                let key_name = key.get().to_owned();
                match item {
                    Item::Value(value) if !matches!(value, Value::InlineTable(_)) => {
                        let label = ui.label(&key_name);
                        if let Some(tip) = &tip {
                            label.on_hover_text(tip);
                        }
                        value_ui(ui, value, &format!("{path}.{key_name}"));
                        ui.end_row();
                    }
                    _ => subtables.push(key_name),
                }
            }
        });
    for key_name in subtables {
        let here = format!("{path}.{key_name}");
        let Some(item) = tab.get_mut(&key_name) else {
            continue;
        };
        match item {
            Item::Table(t) => {
                let tip = comment(t.decor().prefix().and_then(|p| p.as_str()));
                let header = egui::CollapsingHeader::new(format!("[{key_name}]"))
                    .id_salt(&here)
                    .default_open(path.is_empty())
                    .show(ui, |ui| table(ui, t, &here));
                if let Some(tip) = tip {
                    header.header_response.on_hover_text(tip);
                }
            }
            Item::Value(Value::InlineTable(t)) => {
                egui::CollapsingHeader::new(format!("{key_name} {{…}}"))
                    .id_salt(&here)
                    .show(ui, |ui| table(ui, t, &here));
            }
            Item::ArrayOfTables(array) => {
                egui::CollapsingHeader::new(format!("[[{key_name}]] × {}", array.len()))
                    .id_salt(&here)
                    .show(ui, |ui| {
                        for (i, t) in array.iter_mut().enumerate() {
                            let at = format!("{here}[{i}]");
                            egui::CollapsingHeader::new(format!("#{i}"))
                                .id_salt(&at)
                                .show(ui, |ui| table(ui, t, &at));
                        }
                    });
            }
            _ => {}
        }
    }
}

/// A value's field; keeps its decor (the comment after it) when it changes.
fn value_ui(ui: &mut egui::Ui, value: &mut Value, path: &str) {
    let decor = value.decor().clone();
    let mut replaced: Option<Value> = None;
    match value {
        Value::Float(f) => {
            let mut x = *f.value();
            let step = speed(x);
            if ui
                .add(egui::DragValue::new(&mut x).speed(step).max_decimals(6))
                .changed()
            {
                replaced = Some(Value::from(round(x)));
            }
        }
        Value::Integer(i) if is_hex_colour(&i.display_repr(), *i.value()) => {
            // A colour written 0xRRGGBB (sRGB, as `fk_render::Color::hex` reads it).
            let x = *i.value();
            let mut srgb = [(x >> 16) as u8, (x >> 8) as u8, x as u8];
            let response = egui::color_picker::color_edit_button_srgb(ui, &mut srgb);
            response.on_hover_text(format!("0x{x:06x}"));
            let [r, g, b] = srgb.map(i64::from);
            let new = (r << 16) | (g << 8) | b;
            if new != x {
                replaced = format!("0x{new:06x}").parse::<Value>().ok();
            }
        }
        Value::Integer(i) => {
            let mut x = *i.value();
            if ui.add(egui::DragValue::new(&mut x).speed(0.1)).changed() {
                replaced = Some(Value::from(x));
            }
        }
        Value::Boolean(b) => {
            let mut x = *b.value();
            if ui.checkbox(&mut x, "").changed() {
                replaced = Some(Value::from(x));
            }
        }
        Value::String(s) => {
            let mut x = s.value().clone();
            if ui.text_edit_singleline(&mut x).changed() {
                replaced = Some(Value::from(x));
            }
        }
        Value::Array(array) => {
            ui.horizontal_wrapped(|ui| {
                for (i, v) in array.iter_mut().enumerate() {
                    value_ui(ui, v, &format!("{path}[{i}]"));
                }
            });
        }
        Value::Datetime(d) => {
            ui.label(d.value().to_string());
        }
        Value::InlineTable(_) => {}
    }
    if let Some(mut new) = replaced {
        *new.decor_mut() = decor;
        *value = new;
    }
}

/// Whether an integer is a colour: written in hex with six digits.
fn is_hex_colour(repr: &str, value: i64) -> bool {
    let repr = repr.trim();
    repr.len() == 8
        && repr.to_ascii_lowercase().starts_with("0x")
        && (0..=0xff_ffff).contains(&value)
}

/// A drag speed fit for a value's size.
fn speed(x: f64) -> f64 {
    (x.abs() * 0.005).clamp(0.0005, 10.0)
}

/// Six significant digits, so a dragged 0.1 is written 0.1 and not 0.10000000000000002.
fn round(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let digits = 6 - (x.abs().log10().floor() as i32) - 1;
    let k = 10f64.powi(digits.clamp(-300, 300));
    (x * k).round() / k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_edit_keeps_the_comments() {
        let mut doc: DocumentMut = "# the walk\n[walk]\n# how fast\nspeed = 1.5 # m/s\n"
            .parse()
            .unwrap();
        let Item::Value(v) = &mut doc["walk"]["speed"] else {
            panic!()
        };
        let decor = v.decor().clone();
        let mut new = Value::from(round(2.0000000001));
        *new.decor_mut() = decor;
        *v = new;
        assert_eq!(
            doc.to_string(),
            "# the walk\n[walk]\n# how fast\nspeed = 2.0 # m/s\n"
        );
    }

    #[test]
    fn a_hex_colour_is_written_back_in_hex() {
        assert!(is_hex_colour("0xffe6c4", 0xffe6c4));
        assert!(!is_hex_colour("16770756", 16770756));
        let mut doc: DocumentMut = "color = 0xffe6c4 # warm
"
        .parse()
        .unwrap();
        let Item::Value(v) = &mut doc["color"] else {
            panic!()
        };
        let decor = v.decor().clone();
        let mut new: Value = format!("0x{:06x}", 0x102030).parse().unwrap();
        *new.decor_mut() = decor;
        *v = new;
        assert_eq!(
            doc.to_string(),
            "color = 0x102030 # warm
"
        );
    }

    #[test]
    fn comments_become_tooltips() {
        assert_eq!(
            comment(Some("\n# one\n# two\n")).as_deref(),
            Some("one\ntwo")
        );
        assert_eq!(comment(Some("\n")), None);
    }
}
