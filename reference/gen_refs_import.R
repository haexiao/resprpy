# gen_refs_import.R - import_file reference outputs from vignette sample files
# (cp1252 -> UTF-8 converted copies; R on this machine defaults to UTF-8)
suppressPackageStartupMessages(library(respR))
v <- "respR-master/vignettes"
wr <- function(obj, name) {
  write.csv(as.data.frame(obj), file = file.path("../tests/reference", name),
            row.names = FALSE)
}
cat("== Witrox_eg.txt ==\n")
w <- import_file(file.path(v, "Witrox_eg.txt.utf8"))
wr(w, "ref_import_witrox.csv")
cat("nrow:", nrow(w), "ncol:", ncol(w), "\n")
cat("== anch01_raw.txt ==\n")
a <- import_file(file.path(v, "anch01_raw.txt.utf8"))
wr(a, "ref_import_anch01.csv")
cat("nrow:", nrow(a), "ncol:", ncol(a), "\n")
cat("== ACACTB11.csv ==\n")
n <- import_file(file.path(v, "ACACTB11.csv.utf8"))
wr(n, "ref_import_neofox.csv")
cat("nrow:", nrow(n), "ncol:", ncol(n), "\n")
cat("DONE\n")
