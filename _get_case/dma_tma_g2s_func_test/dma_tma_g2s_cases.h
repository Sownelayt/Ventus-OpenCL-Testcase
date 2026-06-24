#ifndef DMA_TMA_G2S_CASES_H
#define DMA_TMA_G2S_CASES_H

int g2s_dma_basic_case_main(int argc, char **argv);
int g2s_copysize_smoke_case_main(int argc, char **argv);
int g2s_bulk_matrix_case_main(int argc, char **argv);
int g2s_tensor_smoke_case_main(int argc, char **argv);
int g2s_tma_descriptor_case_main(void);
int g2s_mixed_async_fence_case_main(void);
int g2s_routing_conflict_case_main(int argc, char **argv);
int g2s_multi_warp_fence_case_main(int argc, char **argv);

#endif /* DMA_TMA_G2S_CASES_H */
