library(reticulate)
library(Rtsne)
library(ggplot2)

# Define the function for t-SNE computation and visualization
perform_tsne_analysis <- function(real_data_path, gen_data_path) {
  
  # Load data using Python
  real_data <- py_run_string(paste0("data = np.load('", real_data_path, "')"))$data
  gen_data <- py_run_string(paste0("data = np.load('", gen_data_path, "')"))$data
  
  # Normalize and prepare data
  normalize_data <- function(data) {
    (data - min(data)) / (max(data) - min(data))
  }
  
  data1_p <- as.data.frame(normalize_data(real_data[, , 1]))
  data2_p <- as.data.frame(normalize_data(real_data[, , 2]))
  data_gen1_p <- as.data.frame(normalize_data(gen_data[, , 1]))
  data_gen2_p <- as.data.frame(normalize_data(gen_data[, , 2]))
  
  # Combine and perform t-SNE
  perform_tsne <- function(data1_p, data_gen1_p) {
    all_data <- rbind(data1_p, data_gen1_p)
    tsne_results <- Rtsne(all_data, perplexity = 10, max_iter = 300, theta = 0.0, check_duplicates = TRUE)
    tsne_df <- data.frame(PC1 = tsne_results$Y[, 1], PC2 = tsne_results$Y[, 2])
    tsne_df$group <- factor(c(rep('real', nrow(data1_p)), rep('gen', nrow(data_gen1_p))),
                            levels = c('real', 'gen'))
    
    # Plotting
    p <- ggplot(tsne_df, aes(x = PC1, y = PC2, color = group)) +
      geom_point(size = 2) +
      stat_ellipse(size = 1) +
      geom_segment(aes(x = -5, xend = 5, y = 0, yend = 0), color = 'red3', size = 0.3) +
      geom_segment(aes(x = 0, xend = 0, y = -5, yend = 5), color = 'red3', size = 0.3) +
      labs(x = 'tSNE1', y = 'tSNE2') +
      theme_minimal() +
      scale_color_manual(values = c('#719CBA', 'grey'))
    
    return(p)
  }
  
  # Perform t-SNE for both sets
  plot1 <- perform_tsne(data1_p, data_gen1_p)
  plot2 <- perform_tsne(data2_p, data_gen2_p)
  
  # Return plots
  list(Plot1 = plot1, Plot2 = plot2)
}

# Display plots
plots <- perform_tsne_analysis(
  real_data_path = "real.npy",
  synthetic_data_path = "diffusion.npy"
  )
print(plots$Plot1)
print(plots$Plot2)
