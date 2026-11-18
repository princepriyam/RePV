# Figure 3(b): areas covered by the assessment sites. Sites within CLUSTER_DEG of each other are merged
# into one area; circle size and number = sites in that area. No per-site labels (too many to read).
#
#   Rscript fig_sites_map.R      -> out/fig_sites_map.pdf
#
# Reads data/sites.csv (python export_fig_data.py). Needs ggplot2, sf, maptiles, tidyterra.
# Basemap tiles are downloaded once and cached in data/tiles.
source(file.path(tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd()), "theme.R"))
suppressPackageStartupMessages({ library(sf); library(maptiles); library(tidyterra) })

BASEMAP     <- "Esri.WorldGrayCanvas"   # or Esri.WorldTerrain, Esri.WorldTopoMap, Esri.WorldImagery
CLUSTER_DEG <- 3                        # sites closer than this (degrees) form one area

sites <- read_data("sites")
sites$area <- cutree(hclust(dist(sites[, c("lon", "lat")]), method = "complete"), h = CLUSTER_DEG)
areas <- aggregate(cbind(lon, lat) ~ area, sites, mean)
areas$n <- as.vector(table(sites$area)[as.character(areas$area)])
stopifnot(sum(areas$n) == nrow(sites))

pts  <- st_as_sf(areas, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)
conus <- st_as_sfc(st_bbox(c(xmin = -125, xmax = -66.5, ymin = 24, ymax = 49.5), crs = 4326)) |> st_transform(3857)
tiles <- get_tiles(conus, provider = BASEMAP, zoom = 5, crop = TRUE,
                   cachedir = file.path(DATA, "tiles"))

p <- ggplot() +
  geom_spatraster_rgb(data = tiles, maxcell = Inf) +
  geom_sf(data = pts, aes(size = n), shape = 21, fill = PAL_CFG[["RAT"]], colour = "white",
          alpha = 0.85, stroke = 0.9) +
  geom_sf_text(data = pts, aes(label = n), size = 4.2, colour = "white", fontface = "bold") +
  scale_size(range = c(7, 17), guide = "none") +
  coord_sf(crs = 3857, datum = NA, expand = FALSE) +
  labs(caption = get_credit(BASEMAP)) +
  theme_void(base_size = 8, base_family = FONT) +
  theme(plot.caption = element_text(size = 6, colour = MUTED))

# caption text for main.tex: number in each circle = PVDAQ sites in that area (sites within 3 deg merged)
save_fig(p, "fig_sites_map", height_mm = 95)
