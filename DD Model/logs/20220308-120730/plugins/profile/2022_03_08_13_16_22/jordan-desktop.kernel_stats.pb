
p
sgemm_32x32x32_TN_vec???*?28??@?2H??b3gradient_tape/sequential_5/dense_16/MatMul/MatMul_1hu  ?A
k
sgemm_32x32x32_TN???*?28î@?4H?OXb1gradient_tape/sequential_5/dense_15/MatMul/MatMulhu  ?A
o
sgemm_32x32x32_NT_vec???*?28??@?4H?PXb1gradient_tape/sequential_5/dense_16/MatMul/MatMulhu  ?A
Z
sgemm_32x32x32_NN_vec???*?28??
@?2H?IXbsequential_5/dense_16/MatMulhu  ?A
V
sgemm_32x32x32_NN???*?28??	@?+H?JXbsequential_5/dense_15/MatMulhu  ?A
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?28??@?)H?+b;sequential_5/dropout_5/dropout/random_uniform/RandomUniformhu  ?B
?
?void gemv2N_kernel<int, int, float, float, float, float, 128, 32, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)0?*?2 8??@?H?0b3gradient_tape/sequential_5/dense_17/MatMul/MatMul_1hu  zB
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28??@?#H?'b/gradient_tape/binary_crossentropy/DynamicStitchhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?%b$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
?void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)3?*?28??@?H?'Xbsequential_5/dense_17/MatMulhu  aB
?
?void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28??@?H?&b7gradient_tape/sequential_5/dense_17/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?$b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?28??@?H?$b7gradient_tape/sequential_5/dense_16/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?!b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?b"Adam/Adam/update/ResourceApplyAdamhu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H? bAdam/gradients/AddN_2hu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?28??@?H?%b7gradient_tape/sequential_5/dense_15/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?"b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b6gradient_tape/binary_crossentropy/weighted_loss/Tile_1hu  ?B
k
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b'binary_crossentropy/logistic_loss/Log1phu  ?B
J
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bAdam/Powhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*?28??@?H?b(ArithmeticOptimizer/AddOpsRewrite_AddN_1hu  HB
?
?void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)?*?28??@?H?Xb1gradient_tape/sequential_5/dense_17/MatMul/MatMulhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?b$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bdense_15/kernel/Regularizer/Sumhu  ?B
?
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/binary_crossentropy/logistic_loss/Reciprocalhu  ?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b2gradient_tape/sequential_5/dropout_5/dropout/Mul_1hu  ?B
u
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?b+sequential_5/dropout_5/dropout/GreaterEqualhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential_5/dense_15/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8??@?H?bdense_16/kernel/Regularizer/Sumhu  ?B
u
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b2gradient_tape/dense_15/kernel/Regularizer/Abs/Signhu  ?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b/gradient_tape/dense_16/kernel/Regularizer/Mul_1hu  ?B
g
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/Exphu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_5hu  ?B
G
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?baddhu  ?B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b0gradient_tape/sequential_5/dropout_5/dropout/Mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b,gradient_tape/sequential_5/dense_16/ReluGradhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?b!dense_15/kernel/Regularizer/Sum_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28??@?H?bdense_16/kernel/Regularizer/Sumhu  HB
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential_5/dense_16/BiasAddhu  ?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b1gradient_tape/dense_16/kernel/Regularizer/Abs/mulhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8??@?H?b!dense_16/kernel/Regularizer/Sum_1hu  ?B
d
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b"sequential_5/dropout_5/dropout/Mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b,gradient_tape/sequential_5/dense_15/ReluGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddNhu  ?B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_15/kernel/Regularizer/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28ߒ@?H?bAssignAddVariableOp_5hu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_1hu  ?B
f
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b$sequential_5/dropout_5/dropout/Mul_1hu  ?B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b"dense_16/kernel/Regularizer/Squarehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_6hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_2hu  ?B
u
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b2gradient_tape/dense_16/kernel/Regularizer/Abs/Signhu  ?B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*?2@8??@?H?bsequential_5/dense_15/Reluhu  ?B
w
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b3gradient_tape/binary_crossentropy/logistic_loss/addhu  ?B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b"dense_15/kernel/Regularizer/Squarehu  ?B
e
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b!binary_crossentropy/logistic_losshu  ?B
L
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?bGreaterhu  ?B
k
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b)gradient_tape/binary_crossentropy/truedivhu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdiv_no_nan_1hu  ?B
a
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_15/kernel/Regularizer/Abshu  ?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b/gradient_tape/dense_15/kernel/Regularizer/Mul_1hu  ?B
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?b.binary_crossentropy/logistic_loss/GreaterEqualhu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_4hu  ?B
g
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/Adam/AssignAddVariableOphu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_2hu  ?B
E
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsubhu  ?B
c
$Sigmoid_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsequential_5/dense_17/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOphu  ?B
g
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/Neghu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b*binary_crossentropy/logistic_loss/Select_1hu  ?B
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b
Adam/Pow_1hu  ?B
]
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*?2@8??@?H?bsequential_5/dense_16/Reluhu  ?B
e
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2@8??@?H?b#sequential_5/dropout_5/dropout/Casthu  ?B
G
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsub_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b6gradient_tape/binary_crossentropy/logistic_loss/Selecthu  ?B
Q
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28??@?H?b
LogicalAndhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_3hu  ?B
y
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b7gradient_tape/binary_crossentropy/logistic_loss/sub/Neghu  ?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b1gradient_tape/dense_15/kernel/Regularizer/Abs/mulhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?b%binary_crossentropy/weighted_loss/Sumhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_4hu  ?B
a
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_16/kernel/Regularizer/Abshu  ?B
E
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bAbshu  ?B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b3gradient_tape/binary_crossentropy/logistic_loss/mulhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28??@?H?b!dense_16/kernel/Regularizer/Sum_1hu  HB
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?badd_1hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_4hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_3hu  ?B
E
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bMulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_3hu  ?B
L
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*?28??@?H?bAdam/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b(binary_crossentropy/logistic_loss/Selecthu  ?B
N
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28??@?H?bAdam/Cast_1hu  ?B
Q
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b
div_no_nanhu  ?B
H
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*?28??@?H?bCasthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_6hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28Ŀ@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_2hu  ?B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_16/kernel/Regularizer/mulhu  ?B
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?badd_2hu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdiv_no_nan_3hu  ?B
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/mulhu  ?B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b!dense_16/kernel/Regularizer/mul_1hu  ?B
n
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28¹@?H?b'binary_crossentropy/weighted_loss/valuehu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28ݸ@?H?bdiv_no_nan_2hu  ?B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b!dense_15/kernel/Regularizer/mul_1hu  ?B
?
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28÷@?H?b@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanhu  ?B
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?bCast_1hu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential_5/dense_17/BiasAddhu  ?B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28??@?H?bCast_3hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_1hu  ?B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b7gradient_tape/binary_crossentropy/logistic_loss/mul/Mulhu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_2hu  ?B
V
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?bGreaterEqualhu  ?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b5gradient_tape/binary_crossentropy/logistic_loss/mul_1hu  ?B
v
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b3binary_crossentropy/weighted_loss/num_elements/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_3hu  ?B
u
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b3gradient_tape/binary_crossentropy/logistic_loss/Neghu  ?B
?
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/binary_crossentropy/logistic_loss/zeros_likehu  ?B
i
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b&gradient_tape/binary_crossentropy/Casthu  ?B
?
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1hu  ?B
?
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?hu  ?B