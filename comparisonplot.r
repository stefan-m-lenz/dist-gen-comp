plotdata <- read.csv("result.tsv", header = TRUE, sep = "\t")

plotdata$modeltype <- toupper(plotdata$modeltype)
plotdata$rmse_odds <- sqrt(plotdata$oddsdelta)


reportNumber <- function(x, nsmall = 2) {
  sapply(x, function(x) {format(x, digits = 2, nsmall = nsmall, decimal.mark = ".")})
}

getQuantiles <- function(plotdata, col) {
  ret <- lapply(list(1,2,5,20), function(nsites) {
      ret <- lapply(list("DBM", "GAN", "MICE", "VAE"), function(modeltype) {
        qs <- quantile(plotdata[plotdata$nsites == nsites & plotdata$modeltype == modeltype, col],
                       probs = c(0.05, 0.5, 0.95))
        qstring <- paste0(reportNumber(qs[2]), " (", reportNumber(qs[1]), " - ", reportNumber(qs[3]), ")")
        ret <- list()
        ret[[modeltype]] <- qstring
        ret
      })
      ret <- unlist(ret, recursive = TRUE)
  })
  ret <- Reduce(rbind, ret)
  rownames(ret) <- c(1,2,5,20)
  ret
}


plotORdist <- function(plotdata) {
  plotdata <- aggregate(rmse_odds ~ datasetindex + modeltype + nsites,
                        data = plotdata, FUN = min)
  par(mfrow = c(2,2))

  ymax <- max(plotdata$rmse_odds[is.finite(plotdata$rmse_odds)])

  ylablogor = "RMSE of ORs"
  boxplot(rmse_odds ~ modeltype,
          data = plotdata[plotdata$nsites == 1, ], ylab = ylablogor,
          xlab = "", ylim = c(0, ymax))
  title("One site")

  boxplot(rmse_odds ~ modeltype,
          plotdata[plotdata$nsites == 2, ], ylab = ylablogor,
          xlab = "", ylim = c(0, ymax))
  title("Two sites")

  boxplot(rmse_odds ~ modeltype,
          plotdata[plotdata$nsites == 5, ], ylab = ylablogor,
          xlab = "", ylim = c(0, ymax))
  title("5 sites")

  boxplot(rmse_odds ~ modeltype,
          plotdata[plotdata$nsites == 20, ], ylab = ylablogor,
          xlab = "", ylim = c(0, ymax))
  title("20 sites")

  getQuantiles(plotdata, "rmse_odds")
}


plotDisclosureMetric <- function(plotdata) {
  plotdata2 <- aggregate(rmse_odds ~ datasetindex + modeltype + nsites,
                        data = plotdata, FUN = min)
  plotdata <- merge(plotdata2, plotdata)

  plotdata$discmetric <- (plotdata$rmse_odds - sqrt(plotdata$oddsdeltatrain))/
    plotdata$rmse_odds

  ymax <- max(plotdata$discmetric[is.finite(plotdata$discmetric)])
  ymin <- min(plotdata$discmetric[is.finite(plotdata$discmetric)])
  ylim <- c(ymin, ymax)

  par(mfrow = c(2,2))

  ylablogor <- "Proportion of overfitting"
  boxplot(discmetric ~ modeltype,
          data = plotdata[plotdata$nsites == 1, ], ylab = ylablogor,
          xlab = "", ylim = ylim)
  title("One site")

  boxplot(discmetric ~ modeltype,
          plotdata[plotdata$nsites == 2, ], ylab = ylablogor,
          xlab = "", ylim = ylim)
  title("Two sites")

  boxplot(discmetric ~ modeltype,
          plotdata[plotdata$nsites == 5, ], ylab = ylablogor,
          xlab = "", ylim = ylim)
  title("5 sites")

  boxplot(discmetric ~ modeltype,
          plotdata[plotdata$nsites == 20, ], ylab = ylablogor,
          xlab = "", ylim = ylim)
  title("20 sites")

  getQuantiles(plotdata, "discmetric")
}

svg(file = "ordist.svg", width = 9, height = 8)
plotORdist(plotdata)
dev.off()

svg(file = "overfitting.svg", width = 9, height = 8)
plotDisclosureMetric(plotdata)
dev.off()

