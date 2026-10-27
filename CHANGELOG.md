# Changelog

Every commit adds a line here, under Unreleased: the pre-commit hook in `.githooks/` refuses a
commit that doesn't, and CI checks the same of every commit pushed. A push to `main` that
releases crates (versions crates.io does not have yet) moves those lines under a heading naming
the date and the crates released; the pre-push hook does it and asks for the push again.

Turn the hooks on once per clone: `git config core.hooksPath .githooks`.

## Unreleased

- fk-render: the layer render (`LayerRender`) is cleared to its clear colour's alpha, no longer
  always opaque, so that a reader may lay what is drawn on it over something else by the alpha.
- fk-physics: `move_through` finds the way out of a surface once a step rather than twice.
- fk-render: `RayMarched::late`, a scene drawn after the image effects, over the image as they
  left it, against the opaque depth (resolved to one sample when multisampled).
- fk-render: a vertex shade (`MeshBuilder::shade`, carried by `Kit`, which numbers its parts),
  bevelled boxes and lumpy subdivided stones (`shapes::bevelled`, `stone`), and a heat map of
  the march's steps (`RayMarching::heat`, `FK_MARCH_HEAT`).
- fk-render: `OtherEyes`, up to three more eyes drawn before the frame as the `Elsewhere` eye
  is, each into an image of its own, shown by `ShowsOther(k)` surfaces: several portals at once.
- fk-render: an `Elsewhere` or `OtherEye` may cast no shadows (`shadows`, sparing its shadow
  pass) and see a far of its own (`far`, nothing drawn or culled for past it).
- fk-flat: `Kit::chunked` cuts a built kit into square chunks of the ground by its triangles'
  middles, each keeping its surfaces, parts and shades, to be culled chunk by chunk.
- fk-render: honest GPU timers: an `offscreen` mark between the shadow pass and the renderers'
  offscreen passes (`GPU_PASSES` eight long), and each extra eye timed on its own (`elsewhere`,
  `eye 1` to `eye 3`).

## 2026-10-27

feuclide 0.2.1

- README: a disclaimer that AI tools were used to split Feuclide out of an experimental game.
- feuclide: 0.2.1, released for its crates.io page to carry the README's disclaimer.

## 2026-10-14

fk-math 0.1.2, fk-shaders 0.1.2, fk-app 0.2.0, fk-scene 0.2.0, fk-quotient 0.2.0, fk-render 0.2.0,
fk-physics 0.2.0, fk-assets 0.2.0, fk-audio 0.2.0, feuclide 0.2.0, fk-sky 0.2.0, fk-ui 0.2.0,
fk-flat 0.2.0

- fk-render: the game's look leaves the engine (the ink, the datamosh, the lids and the
  afterimage, the lenses, the streaks, the posterize, the tunnel, the sky image, the star
  figure); `ImageEffect`, `GradeShader` (the `fk::grade` hook), `SkyFigure` and `sdf_alpha`
  let a game add them back.
- fk-render: `Settle`, the image settling over frames; interleaved and coarse ray marching;
  extruded, cone, boulder, smooth and up-quad shapes, box UVs; `MeshBuilder::framed`, a window
  onto the other eye framing its whole image; a frame never waits for a field's pipelines
  (`FieldsBuilt`).
- fk-app: gamepads, `Progress` for loading steps, tuning files reloaded on save
  (`FK_TUNING_DIR`), `TomlFile`, `Handover::carry`, `StickLatch`.
- fk-physics: cast, normal and curvature over any distance field.
- fk-math: `smoothstep`, `rand`, value noise and `sdf`; fk-shaders: `fk::sdf_lib`, `wind_gust`
  and `swayed_by` in `fk::displace`.
- fk-audio: directional emitters, `Takes`, `synth`, a walker's gait.
- fk-sky (new): the day and the night, a clock and the colours of every hour.
- fk-ui (new): the frame rate, the list of keys, a pointer, a debug console's core, menu pages.
- fk-flat (new): flat worlds, standing on a ground, free spots, kits, chunks, height-field land.
- fk-quotient, fk-scene, fk-assets, feuclide: released against the new fk-app and fk-render.
- Every crate in the README table links to crates.io, or to its folder for the unpublished.
- This changelog, with hooks and a CI check that every commit adds a line to it.

## 2026-10-13

fk-math 0.1.1, fk-geometry 0.1.1, fk-geometry-euclidean 0.1.1, fk-geometry-hyperbolic 0.1.1,
fk-geometry-spherical 0.1.1, fk-quotient 0.1.1, fk-scene 0.1.1, fk-shaders 0.1.1,
fk-render 0.1.1, fk-physics 0.1.1, fk-assets 0.1.1, fk-app 0.1.1, fk-audio 0.1.1,
feuclide 0.1.2

- Each crate has a README of its own, with the part of Bevy it stands in for.
- CI publishes every crate whose version is not on crates.io yet.

## 2026-10-10

feuclide 0.1.1

- The README's showcase.

## 2026-10-08

All crates 0.1.0

- First release.
