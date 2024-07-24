library(reticulate)
library(dtw)
library(ggplot2)
library(plyr)

# DTW distance calculation function
dtw_distance <- function(matrix1, matrix2) {
  n1 <- nrow(matrix1)
  n2 <- nrow(matrix2)
  dist_matrix <- matrix(0, nrow = n1, ncol = n2)
  
  for (i in 1:n1) {
    for (j in 1:n2) {
      alignment <- dtw(matrix1[i, , drop = FALSE], matrix2[j, , drop = FALSE], keep.internals = TRUE)
      dist_matrix[i, j] <- alignment$distance
    }
  }
  return(dist_matrix)
}

# Main analysis function
analyze_dtw_distances <- function(real_data_path, synthetic_data_path) {
  
  # Load real and synthetic data using Python
  real_data <- py_run_string(paste0("data = np.load('", real_data_path, "')"))$data
  synthetic_data <- py_run_string(paste0("data = np.load('", synthetic_data_path, "')"))$data
  
  # Convert to data frames and normalize
  process_data <- function(data, index) {
    df <- as.data.frame(data[, , index])
    (df - min(df)) / (max(df) - min(df))
  }
  
  data1_p <- process_data(real_data, 1)
  data2_p <- process_data(real_data, 2)
  data_gen1 <- process_data(synthetic_data, 1)
  data_gen2 <- process_data(synthetic_data, 2)
  
  # Compute DTW distances
  c1 <- unlist(dtw_distance(data1_p, data_gen1))
  c2 <- unlist(dtw_distance(data2_p, data_gen2))
  
  # Aggregate results and plot
  c <- data.frame(cal = c(c1, c2), f = rep(c('f1', 'f2'), each = length(c1)))
  mu <- ddply(c, "f", summarise, grp.mean = median(cal))
  
  p <- ggplot(c, aes(x = cal, fill = f)) +
    geom_density(alpha = 0.4) +
    geom_vline(data = mu, aes(xintercept = grp.mean, color = f), linetype = "dashed", size = 0.5) +
    xlim(0, 14) +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_fill_manual(values = c('#719CBA', '#719CBA')) + 
    scale_color_manual(values = c('black', 'black'))
  
  return(p)
}

# Display plots
plot <- analyze_dtw_distances(
  real_data_path = "real.npy",
  synthetic_data_path = "diffusion.npy"
)
print(plot)
