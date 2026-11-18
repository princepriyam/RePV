# Figure 5: effect of model refinement and of the irradiance input. Replaces Table 8 (val_delta.tex).
#   (a) median change per refinement-path step (same definition as val_delta.tex)
#   (b) per-site NRMSE reduction: refinement path on ERA5 against the ERA5 -> NSRDB input change
#   (c) optional: selected ablations, feature added to MOD against removed from SYS (NSRDB)
#
#   Rscript fig_refinement.R     -> out/fig_refinement_{steps,vs_input,ablation}.pdf
#
# Reads data/scores.csv, data/sites.csv, data/path.csv, data/ablation.csv (python export_fig_data.py).
# Needs ggplot2, patchwork.
source(file.path(tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd()), "theme.R"))
suppressPackageStartupMessages(library(patchwork))

INCLUDE_C    <- TRUE                                   # FALSE: panels (a) and (b) only
ABLATION_KEYS <- c("Iam", "Trig", "Pres", "Snow")      # keys of FEATURES in make_validation_tables.py
ACT_TOL      <- 0.1                                    # |delta b| threshold of the site count (points)

sc    <- read_data("scores")
sites <- read_data("sites")
path  <- read_data("path")
abl   <- read_data("ablation")
ind   <- sites$site[sites$group == "ind"]
sc    <- subset(sc, site %in% ind)
key   <- function(d) paste(d$site, d$model, d$input)
lookup <- function(site, model, input, col) sc[[col]][match(paste(site, model, input), key(sc))]

# ── (a) refinement-path steps ──────────────────────────────────────────────
order <- c("RAT", paste0("P", 0:10))
model_of <- setNames(path$model, path$step)
STEP_LAB <- c(P0 = "RAT → MOD", P1 = "P1  Site-pressure air mass", P2 = "P2  Diffuse IAM",
              P3 = "P3  Terrain horizon", P4 = "P4  Loss placement, Pac0", P5 = "P5  Optical soiling",
              P6 = "P6  AC capacity limit", P7 = "P7  Snow-depth condition", P8 = "P8  6 h snowfall trigger",
              P9 = "P9  Snow-dependent albedo", P10 = "P10  Row geometry (SYS-PVW)")

steps <- do.call(rbind, lapply(2:length(order), function(i) {
  st <- order[i]; m <- model_of[[st]]; m0 <- model_of[[order[i - 1]]]
  do.call(rbind, lapply(c("point", "nsrdb"), function(inp) {
    db <- lookup(ind, m, inp, "bias") - lookup(ind, m0, inp, "bias")
    dn <- lookup(ind, m, inp, "nrmse") - lookup(ind, m0, inp, "nrmse")
    ok <- !is.na(db)
    data.frame(step = st, input = INPUT_LAB[[inp]], metric = c("b", "nrmse"),
               med = c(median(db[ok]), median(dn[ok])), act = sum(abs(db[ok]) >= ACT_TOL), n = sum(ok))
  }))
}))
# table layout: one row per step, a header row per section; x = column centre
row_y <- c(P0 = 1, setNames(3:12, paste0("P", 1:10)))            # rows 0 and 2 are section headers
col_x <- c(b.ERA5 = 1, b.NSRDB = 2, nrmse.ERA5 = 3.3, nrmse.NSRDB = 4.3)
steps$y <- -row_y[steps$step]
steps$x <- col_x[paste(steps$metric, steps$input, sep = ".")]
steps$rel  <- with(steps, med / ave(abs(med), metric, FUN = max))  # fill: sign and size within the metric
steps$zero <- abs(steps$med) < 0.005
fmt <- function(v) ifelse(v == 0, "0.00", ifelse(abs(v) < 0.005, "<0.01", sub("-", "−", sprintf("%+.2f", v))))
steps$txt <- ifelse(steps$zero & steps$med != 0, "<0.01", fmt(steps$med))
cnt <- subset(steps, metric == "b")
cnt$x <- ifelse(cnt$input == "ERA5", 5.6, 6.6)
lab <- data.frame(step = names(row_y), y = -row_y, txt = STEP_LAB[names(row_y)])
sec <- data.frame(y = c(0, -2), txt = c("REFERENCE COMPARISON", "CONTROLLED MOD → SYS-PVW REFINEMENT"))
hd2 <- data.frame(x = c(col_x, 5.6, 6.6), txt = rep(c("ERA5", "NSRDB"), 3))
hd1 <- data.frame(x = c(1.5, 3.8, 6.1), txt = c("Signed bias", "Hourly NRMSE", "Sites with |Δb| ≥ 0.1 pp"))
X0 <- -2.2                                                        # left edge of the step labels
TFONT <- "Arial"                                                  # has the arrow and ≥ glyphs; the device default does not

pa <- ggplot() +
  geom_tile(data = steps, aes(x, y, fill = ifelse(zero, NA, rel)), width = 0.96, height = 0.86) +
  geom_text(data = steps, aes(x, y, label = txt, colour = zero), family = TFONT, size = 2.4) +
  geom_text(data = cnt, aes(x, y, label = paste0(act, "/", n)), family = TFONT, size = 2.3, colour = MUTED) +
  geom_text(data = lab, aes(X0, y, label = txt), hjust = 0, family = TFONT, size = 2.5, colour = INK) +
  geom_text(data = sec, aes(X0, y, label = txt), hjust = 0, family = TFONT, size = 2.3, colour = MUTED, fontface = "bold") +
  geom_text(data = hd2, aes(x, 1.1, label = txt), family = TFONT, size = 2.3, colour = MUTED) +
  geom_text(data = hd1, aes(x, 1.9, label = txt), family = TFONT, size = 2.5, colour = INK, fontface = "bold") +
  annotate("segment", x = c(0.55, 2.85, 5.15), xend = c(2.45, 4.75, 7.05), y = 1.5, yend = 1.5,
           colour = MUTED, linewidth = 0.3) +
  annotate("text", x = X0, y = 2.8, hjust = 0, family = TFONT, size = 2.7, colour = INK, fontface = "bold",
           label = "Median paired step effect, percentage points") +
  scale_fill_gradient2(low = NEG, mid = "#f7f6f2", high = POS, midpoint = 0, limits = c(-1, 1),
                       na.value = "#f7f6f2", guide = "none") +
  scale_colour_manual(values = c(`TRUE` = MUTED, `FALSE` = INK), guide = "none") +
  coord_cartesian(xlim = c(X0, 7.1), ylim = c(-12.5, 3.1), expand = FALSE) +
  theme_void(base_family = FONT)

# ── (b) model detail against irradiance input ──────────────────────────────
nr <- function(m, inp) lookup(ind, m, inp, "nrmse")
gain <- data.frame(site = ind,
                   rat_sys = nr("homer", "point") - nr("current", "point"),
                   mod_sys = nr("thesis", "point") - nr("current", "point"),
                   era_nsrdb = nr("current", "point") - nr("current", "nsrdb"))
gain <- gain[complete.cases(gain), ]
n_beat <- sum(gain$era_nsrdb > gain$rat_sys)
gl <- rbind(data.frame(cmp = "RAT to SYS-PVW (ERA5)", v = gain$rat_sys),
            data.frame(cmp = "MOD to SYS-PVW (ERA5)", v = gain$mod_sys),
            data.frame(cmp = "ERA5 to NSRDB (SYS-PVW)", v = gain$era_nsrdb))
gl$cmp <- factor(gl$cmp, levels = rev(unique(gl$cmp)))
gm <- aggregate(v ~ cmp, gl, median)

pb <- ggplot(gl, aes(v, cmp)) +
  geom_vline(xintercept = 0, colour = MUTED, linewidth = 0.3) +
  geom_point(position = position_jitter(height = 0.12, width = 0, seed = 1), size = 1.1,
             colour = PAL_CFG[["SYS"]], alpha = 0.7) +
  geom_point(data = gm, shape = 124, size = 5, colour = "#1f1f1e") +
  geom_text(data = gm, aes(label = sprintf("%.1f", v)), vjust = -1.1, size = 2.3, colour = INK) +
  annotate("text", x = Inf, y = 3.45, hjust = 1, vjust = 0, size = 2.2, colour = INK,
           label = sprintf("input change > full path at %d of %d sites", n_beat, nrow(gain))) +
  labs(x = "NRMSE reduction per site (points; positive = closer to measured)", y = NULL) +
  coord_cartesian(clip = "off") +
  theme_paper(7.5) + theme(panel.grid.major.y = element_blank(),
                           panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                           axis.text.y = element_text(colour = INK))

# ── (c) ablations: added to MOD against removed from SYS (NSRDB) ───────────
pc <- NULL
if (INCLUDE_C) {
  a <- subset(abl, key %in% ABLATION_KEYS)
  a$med <- mapply(function(w, o) {
    d <- lookup(ind, w, "nsrdb", "bias") - lookup(ind, o, "nsrdb", "bias")
    median(d, na.rm = TRUE)
  }, a$with_model, a$without_model)
  a$feature <- factor(a$feature, levels = rev(unique(a$feature[match(ABLATION_KEYS, a$key)])))
  pc <- ggplot(a, aes(med, feature)) +
    geom_vline(xintercept = 0, colour = MUTED, linewidth = 0.3) +
    geom_line(aes(group = feature), colour = MUTED, linewidth = 0.4) +
    geom_point(aes(shape = direction, fill = direction), size = 2, colour = INK, stroke = 0.4) +
    scale_shape_manual(values = c(`Added to MOD` = 21, `Removed from SYS` = 23), name = NULL) +
    scale_fill_manual(values = c(`Added to MOD` = PAL_CFG[["MOD"]], `Removed from SYS` = PAL_CFG[["SYS"]]), name = NULL) +
    labs(x = "Median Δb of the feature, NSRDB (points)", y = NULL) +
    theme_paper(7.5) + theme(legend.position = "bottom", panel.grid.major.y = element_blank(),
                             panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                             axis.text.y = element_text(colour = INK))
}

# one PDF per panel: the combined figure was too dense to read
save_fig(pa, "fig_refinement_steps", height_mm = 80)
save_fig(pb, "fig_refinement_vs_input", height_mm = 60)
if (INCLUDE_C) save_fig(pc, "fig_refinement_ablation", height_mm = 60)

# caption text for main.tex: (a) medians across the independent sites, each step measured from the step
# before it (P0 from RAT); colour = sign and size relative to the largest median of the same metric;
# right column = sites at which |Δb| >= 0.1 points (a feature can be active below this threshold)
