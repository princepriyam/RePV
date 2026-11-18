# Shared palette, theme and output helper of the MethodsX R figures.
# Sourced by fig_*.R. Colours of RAT/MOD/SYS/PVGIS match fig_validation.pdf and the TikZ figures
# (define the same hex values with \definecolor in LaTeX if the TikZ figures use them).

suppressPackageStartupMessages({
  library(ggplot2)
})

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
DATA <- file.path(here, "data")
OUTD <- file.path(here, "out")
dir.create(OUTD, showWarnings = FALSE)

# configurations (same slots as the current fig_validation.pdf)
PAL_CFG <- c(RAT = "#2a78d6", MOD = "#eb6834", SYS = "#1baf7a", PVGIS = "#eda100")
SHP_CFG <- c(RAT = 21, MOD = 22, SYS = 23, PVGIS = 24)
# evidence classes (not reused from the configuration palette)
PAL_CLS <- c(A = "#3d3a8c", B = "#b8487a", C = "#3a3a38")
SHP_CLS <- c(A = 21, B = 24, C = 22)
FILL_CLS <- c(A = "#3d3a8c", B = "#b8487a", C = "white")

INK <- "#3a3a38"; MUTED <- "#8a8984"; GRID <- "#e6e5e0"
NEG <- "#2a78d6"; POS <- "#d9622b"            # diverging ends for signed changes

TEXTWIDTH_MM <- 160                          # A4, 25 mm margins (main.tex geometry)
FONT <- ""                                   # e.g. "Latin Modern Sans" if installed; "" = device default

theme_paper <- function(base_size = 8) {
  theme_minimal(base_size = base_size, base_family = FONT) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
          axis.text = element_text(colour = MUTED),
          axis.title = element_text(colour = INK),
          strip.text = element_text(colour = INK, face = "bold"),
          legend.text = element_text(colour = INK),
          legend.title = element_text(colour = INK),
          plot.tag = element_text(face = "bold", size = base_size + 1))
}

save_fig <- function(p, name, height_mm, width_mm = TEXTWIDTH_MM) {
  out <- file.path(OUTD, paste0(name, ".pdf"))
  ggsave(out, p, width = width_mm, height = height_mm, units = "mm", device = cairo_pdf)
  message("-> ", out)
}

read_data <- function(name) read.csv(file.path(DATA, paste0(name, ".csv")), stringsAsFactors = FALSE)

# series key as in bootstrap.parquet: "<model>|<input>"; PVGIS is "pvgis|PVGIS-ERA5"
CFG_MODEL <- c(homer = "RAT", thesis = "MOD", current = "SYS")
INPUT_LAB <- c(point = "ERA5", nsrdb = "NSRDB", midc = "Station")
