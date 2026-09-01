# Dump ALL respR namespace objects (functions + regex/constant objects) + marelac functions
suppressPackageStartupMessages(library(respR))
outdir <- "X:/Rtools/resprpy/reference/r_src"
ns <- asNamespace("respR")
allf <- ls(ns, all.names = TRUE)
n <- 0
for (f in allf) {
  obj <- get(f, envir = ns)
  txt <- tryCatch(deparse(obj, control = "all"), error = function(e) NULL)
  if (!is.null(txt)) {
    writeLines(txt, file.path(outdir, paste0(f, ".R")))
    n <- n + 1
  }
}
cat("dumped", n, "objects\n")

# marelac: the 6 physical functions used by respR
suppressPackageStartupMessages(library(marelac))
mout <- "X:/Rtools/resprpy/reference/r_src/marelac"
dir.create(mout, recursive = TRUE, showWarnings = FALSE)
mfs <- c("molvol", "molweight", "gas_satconc", "sw_dens", "vapor", "atmComp")
for (f in mfs) {
  fn <- get(f, envir = asNamespace("marelac"))
  txt <- deparse(fn, control = "all")
  writeLines(txt, file.path(mout, paste0(f, ".R")))
  cat("marelac:", f, length(txt), "lines\n")
}
