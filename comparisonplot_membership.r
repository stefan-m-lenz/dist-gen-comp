membership_plotdata <- read.csv("results_membership.tsv",
                                header = TRUE, sep = "\t")

membership_plotdata$precision <- membership_plotdata$true_pos /
  ( membership_plotdata$true_pos + membership_plotdata$false_pos)

membership_plotdata$sensitivity <- membership_plotdata$true_pos /
  ( membership_plotdata$true_pos + membership_plotdata$false_neg)


plotMembershipMetric <- function(membership_plotdata, what) {
  ylim <- c(0,1)

  par(mfrow = c(4,1), mar = c(4, 5, 4, 2))
  modeltypes <- sort(toupper(unique(membership_plotdata$modeltype)))
  ndistances <- 7
  atarg <- unlist(Map(function(i) {((i-1)*ndistances+i):(i*ndistances+(i-1))}, 1:5))
  boxplotforsite <- function(nsites, title, what) {
    ylablogor <- paste(toupper(substring(what, 1,1)), substring(what, 2),
                       sep="", collapse=" ")
    boxplot(as.formula(paste0(what, " ~  distance + modeltype")),
            data = membership_plotdata[membership_plotdata$nsites == nsites, ],
            ylab = ylablogor, xlab = "", ylim = ylim, at = atarg,
            names = as.character(rep(c(0, 2, 3, 5, 6, 8, 10), 5)),
            yaxt = 'n',
            cex.lab = 1.5)
    axis(side = 2, at = c(0, 0.25, 0.5, 0.75, 1),
         labels = c("0", "", "0.5", "", "1"))

    for (i in 1:length(modeltypes)) {
      mtext(modeltypes[i], line = 0,
            at = 3 + (i-1) * ndistances + i,
            cex = 1)
    }
    title(title, line = 2, cex.main = 1.5)
  }

  boxplotforsite(1, "One site", what)
  boxplotforsite(2, "Two sites", what)
  boxplotforsite(5, "5 sites", what)
  boxplotforsite(20, "20 sites", what)

}


pdf(file = "membership_precision.pdf", width = 8.8, height = 7.5)
plotMembershipMetric(membership_plotdata, "precision")
dev.off()

pdf(file = "membership_sensitivity.pdf", width = 8.8, height = 7.5)
plotMembershipMetric(membership_plotdata, "sensitivity")
dev.off()
