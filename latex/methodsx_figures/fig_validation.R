# Figure: per-site |b| and NRMSE of the aggregate baseline and RePV against measured AC output,
# independent validation sites only. Grey lines join the same site across the first three series:
# baseline -> RePV on ERA5 isolates the conversion chain, RePV ERA5 -> RePV NSRDB the irradiance input.
# PVGIS-ERA5 is set apart as the external benchmark. Black bars are medians (values in Table 1).
#
#   Rscript fig_validation.R     -> out/fig_validation.pdf
#
# Reads data/scores.csv and data/sites.csv (python export_fig_data.py). Needs ggplot2, patchwork.
source(file.path(tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd()), "theme.R"))
suppressPackageStartupMessages(library(patchwork))

sc    <- read_data("scores")
sites <- read_data("sites")
ind   <- sites$site[sites$group == "ind"]
golden <- sites$site[sites$place == "Golden"]

# series key in scores.csv: "<model>|<input>"; homer = aggregate baseline, current = RePV
SER <- data.frame(
  series = c("homer|point", "current|point", "current|nsrdb", "pvgis|PVGIS-ERA5"),
  x      = c(1, 2, 3, 4.35),
  ser    = c("Baseline", "RePV", "RePV", "PVGIS"),
  lab    = c("Rating model\nERA5", "RePV\nERA5", "RePV\nNSRDB", "PVGIS\nPVGIS-ERA5"))
PAL_SER <- c(Baseline = PAL_CFG[["RAT"]], RePV = PAL_CFG[["SYS"]], PVGIS = PAL_CFG[["PVGIS"]])

d <- merge(subset(sc, site %in% ind, c(site, series, bias, nrmse)), SER, by = "series")
stopifnot(all(table(d$series) == length(ind)))

# fixed horizontal offset per site, identical in every series, so joined points do not overplot
set.seed(7)
off <- setNames(sample(seq(-0.13, 0.13, length.out = length(ind))), ind)
d$xj <- d$x + off[d$site]
CLU <- c(sprintf("Golden sites (%d)", sum(ind %in% golden)), sprintf("Other sites (%d)", sum(!ind %in% golden)))
d$cluster <- factor(ifelse(d$site %in% golden, CLU[1], CLU[2]), levels = CLU)

panel <- function(col, ytitle) {
  d$y <- if (col == "bias") abs(d$bias) else d[[col]]
  med <- aggregate(y ~ x + ser, d, median)
  top <- max(d$y) * 1.16
  ggplot() +
    annotate("segment", x = 3.675, xend = 3.675, y = 0, yend = top, colour = GRID, linewidth = 0.4, linetype = "22") +
    geom_point(data = d, aes(xj, y, fill = ser, shape = cluster), size = 1.7, stroke = 0.25, colour = "white") +
    geom_segment(data = med, aes(x = x - 0.24, xend = x + 0.24, y = y, yend = y), colour = "#1f1f1e", linewidth = 0.7,
                 lineend = "round") +
    annotate("text", x = c(1.5, 2.5, 4.35), y = top, vjust = 1, size = 2.3, colour = MUTED,
             label = c("conversion\nto power", "irradiance\ninput", "benchmark"), lineheight = 0.9) +
    scale_shape_manual(values = setNames(c(24, 21), CLU), name = NULL,
                       guide = guide_legend(override.aes = list(fill = "#8a8984", size = 2))) +
    scale_fill_manual(values = PAL_SER, guide = "none") +
    scale_x_continuous(breaks = SER$x, labels = SER$lab, expand = expansion(add = 0.35)) +
    scale_y_continuous(limits = c(0, top), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = ytitle) +
    theme_paper(8) +
    theme(axis.text.x = element_text(colour = INK, lineheight = 0.9),
          legend.position = "bottom", legend.key.width = unit(5, "mm"),
          legend.margin = margin(0, 0, 0, 0))
}

p <- (panel("bias", "|b| (%)") | panel("nrmse", "NRMSE (%)")) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(legend.position = "bottom", legend.box.spacing = unit(0, "mm"),
        legend.title = element_text(colour = MUTED, size = 7))

save_fig(p, "fig_validation", height_mm = 72)
