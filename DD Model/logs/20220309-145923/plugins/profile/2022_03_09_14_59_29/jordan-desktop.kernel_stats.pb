
W
sgemm_32x32x32_NN_vec???*?28ߧ@?5H?OXbsequential/dense_1/MatMulhu  ?A
l
sgemm_32x32x32_NT_vec???*?28??@?6H?NXb.gradient_tape/sequential/dense_1/MatMul/MatMulhu  ?A
f
sgemm_32x32x32_TN???*?28??@?4H?OXb,gradient_tape/sequential/dense/MatMul/MatMulhu  ?A
Q
sgemm_32x32x32_NN???*?28??@?-H?NXbsequential/dense/MatMulhu  ?A
l
sgemm_32x32x32_TN_vec???*?28??@?2H?Qb0gradient_tape/sequential/dense_1/MatMul/MatMul_1hu  ?A
?
?void gemv2N_kernel<int, int, float, float, float, float, 128, 32, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)0?*?2 8??@?%H?.b0gradient_tape/sequential/dense_2/MatMul/MatMul_1hu  zB
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*?28??@?)H?,b7sequential/dropout/dropout/random_uniform/RandomUniformhu  ?B
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28??@?#H?*b/gradient_tape/binary_crossentropy/DynamicStitchhu  ?B
?
?void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)3?*?28??@?H?&Xbsequential/dense_2/MatMulhu  aB
?
?void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28??@?H?*b4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?&b$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?#b$Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?b+sequential/batch_normalization/moments/meanhu  ?B
?
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H? b8sequential/batch_normalization/moments/SquaredDifferencehu  ?B
|
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/sequential/batch_normalization_1/moments/subhu  ?B
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization/batchnorm/add_1hu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization/batchnorm/mul_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?b-sequential/batch_normalization_1/moments/meanhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bBgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Mulhu  ?B
z
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b8gradient_tape/sequential/batch_normalization/moments/subhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization/batchnorm/mul_1/Mulhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28߿@?H?b1sequential/batch_normalization_1/moments/variancehu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?b/sequential/batch_normalization/moments/variancehu  ?B
?
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:sequential/batch_normalization_1/moments/SquaredDifferencehu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?%b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b0sequential/batch_normalization_1/batchnorm/mul_1hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28ݵ@?H?$b"Adam/Adam/update/ResourceApplyAdamhu  ?B
k
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?!b'binary_crossentropy/logistic_loss/Log1phu  ?B
t
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b0sequential/batch_normalization_1/batchnorm/add_1hu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?28??@?H?b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?bBgradient_tape/sequential/batch_normalization_1/moments/BroadcastTohu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b6gradient_tape/binary_crossentropy/weighted_loss/Tile_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b@gradient_tape/sequential/batch_normalization/moments/BroadcastTohu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?"b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?28??@?H?b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?bDgradient_tape/sequential/batch_normalization_1/moments/BroadcastTo_1hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?"b$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?bBgradient_tape/sequential/batch_normalization/moments/BroadcastTo_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*?28??@?H?b(ArithmeticOptimizer/AddOpsRewrite_AddN_1hu  HB
?
?void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)?*?28??@?H?Xb.gradient_tape/sequential/dense_2/MatMul/MatMulhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/add_1/Sum_1hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?!b$Adam/Adam/update_6/ResourceApplyAdamhu  ?B
J
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bAdam/Powhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?b+sequential/batch_normalization/moments/meanhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/add_1/Sum_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/add_1/Sum_1hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?b$Adam/Adam/update_7/ResourceApplyAdamhu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_6hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bdense/kernel/Regularizer/Sumhu  ?B
?
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b>gradient_tape/sequential/batch_normalization/moments/truediv_1hu  ?B
?
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization_1/moments/truediv_1hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@?H?b$Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Sum_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28??@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Sum_1hu  ?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?b'sequential/dropout/dropout/GreaterEqualhu  ?B
?
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/binary_crossentropy/logistic_loss/Reciprocalhu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_4hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/add_1/Sum_1hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?b-sequential/batch_normalization_1/moments/meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_5hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8¼@?H?bdense_1/kernel/Regularizer/Sumhu  ?B
g
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/Exphu  ?B
r
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b/gradient_tape/dense/kernel/Regularizer/Abs/Signhu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.gradient_tape/dense_1/kernel/Regularizer/Mul_1hu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential/dense/BiasAddhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?b1sequential/batch_normalization_1/moments/variancehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddNhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28ߪ@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Mul_1hu  ?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b8gradient_tape/sequential/batch_normalization/moments/Mulhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Sum_1hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Sum_1hu  ?B
~
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b<gradient_tape/sequential/batch_normalization/moments/truedivhu  ?B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/sequential/batch_normalization_1/moments/Mulhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)?!*  28??@?H?b/sequential/batch_normalization/moments/variancehu  ?B
?
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b>gradient_tape/sequential/batch_normalization_1/moments/truedivhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_5hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bdense/kernel/Regularizer/Sum_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_2hu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential/dense_1/BiasAddhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsequential/dropout/dropout/Mulhu  ?B
G
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28ޚ@?H?baddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b)gradient_tape/sequential/dense_1/ReluGradhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28??@?H?bdense_1/kernel/Regularizer/Sumhu  HB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28ޑ@?H?b0gradient_tape/dense_1/kernel/Regularizer/Abs/mulhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?2 8??@?H?b dense_1/kernel/Regularizer/Sum_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b'gradient_tape/sequential/dense/ReluGradhu  ?B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/sequential/batch_normalization/moments/mul_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bBgradient_tape/sequential/batch_normalization_1/batchnorm/RsqrtGradhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_2hu  ?B
X
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*?2@8??@?H?bsequential/dense/Reluhu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.gradient_tape/sequential/dropout/dropout/Mul_1hu  ?B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b!dense_1/kernel/Regularizer/Squarehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b.sequential/batch_normalization/AssignMovingAvghu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b sequential/dropout/dropout/Mul_1hu  ?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b<gradient_tape/sequential/batch_normalization_1/moments/mul_1hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28߁@?H?bmul_4hu  ?B
t
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b1gradient_tape/dense_1/kernel/Regularizer/Abs/Signhu  ?B
E
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsubhu  ?B
t
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b2sequential/batch_normalization/AssignMovingAvg/subhu  ?B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense/kernel/Regularizer/Squarehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/Adam/AssignAddVariableOphu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAdam/gradients/AddN_5hu  ?B
L
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?bGreaterhu  ?B
^
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense/kernel/Regularizer/Abshu  ?B
r
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization/batchnorm/Rsqrthu  ?B
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?b.binary_crossentropy/logistic_loss/GreaterEqualhu  ?B
p
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b,sequential/batch_normalization/batchnorm/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_6hu  ?B
`
$Sigmoid_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsequential/dense_2/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b@gradient_tape/sequential/batch_normalization/batchnorm/RsqrtGradhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b,gradient_tape/dense/kernel/Regularizer/Mul_1hu  ?B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense/kernel/Regularizer/mulhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b,gradient_tape/sequential/dropout/dropout/Mulhu  ?B
e
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b!binary_crossentropy/logistic_losshu  ?B
k
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b)gradient_tape/binary_crossentropy/truedivhu  ?B
w
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b3gradient_tape/binary_crossentropy/logistic_loss/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_4hu  ?B
g
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/Neghu  ?B
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization_1/batchnorm/addhu  ?B
E
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bMulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOphu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b*binary_crossentropy/logistic_loss/Select_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b6gradient_tape/binary_crossentropy/logistic_loss/Selecthu  ?B
Z
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*?2@8??@?H?bsequential/dense_1/Reluhu  ?B
G
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bsub_1hu  ?B
E
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bAbshu  ?B
?
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanhu  ?B
`
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_1/kernel/Regularizer/Abshu  ?B
a
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?2@8??@?H?bsequential/dropout/dropout/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b0sequential/batch_normalization_1/AssignMovingAvghu  ?B
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b
Adam/Pow_1hu  ?B
y
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b7gradient_tape/binary_crossentropy/logistic_loss/sub/Neghu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization_1/batchnorm/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b0sequential/batch_normalization/AssignMovingAvg_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_3hu  ?B
Q
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28??@?H?b
LogicalAndhu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdiv_no_nan_1hu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.gradient_tape/dense/kernel/Regularizer/Abs/mulhu  ?B
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?badd_2hu  ?B
?
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b>gradient_tape/sequential/batch_normalization/batchnorm/sub/Neghu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?bAssignAddVariableOp_1hu  ?B
?
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization_1/batchnorm/sub/Neghu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@?H?b2sequential/batch_normalization_1/AssignMovingAvg_1hu  ?B
L
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*?28??@?H?bAdam/addhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b,sequential/batch_normalization/batchnorm/mulhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28??@?H?bsequential/dense_2/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_2hu  ?B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b4sequential/batch_normalization_1/AssignMovingAvg/subhu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdiv_no_nan_3hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_4hu  ?B
t
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b0sequential/batch_normalization_1/batchnorm/Rsqrthu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?b%binary_crossentropy/weighted_loss/Sumhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_3hu  ?B
H
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*?28??@?H?bCasthu  ?B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b4sequential/batch_normalization/AssignMovingAvg_1/subhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28??@?H?bSum_3hu  ?B
N
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28??@?H?bAdam/Cast_1hu  ?B
g
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/subhu  ?B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b3gradient_tape/binary_crossentropy/logistic_loss/mulhu  ?B
x
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b6sequential/batch_normalization_1/AssignMovingAvg_1/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@?H?b(binary_crossentropy/logistic_loss/Selecthu  ?B
n
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b,sequential/batch_normalization/batchnorm/subhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ŀ@?H?b dense_1/kernel/Regularizer/Sum_1hu  HB
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b%binary_crossentropy/logistic_loss/mulhu  ?B
Q
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28߾@?H?b
div_no_nanhu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28޾@?H?bmul_6hu  ?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b4sequential/batch_normalization_1/AssignMovingAvg/mulhu  ?B
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?badd_1hu  ?B
n
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b'binary_crossentropy/weighted_loss/valuehu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_1hu  ?B
v
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b3binary_crossentropy/weighted_loss/num_elements/Casthu  ?B
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?bCast_1hu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b dense_1/kernel/Regularizer/mul_1hu  ?B
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28??@?H?bCast_3hu  ?B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization_1/batchnorm/subhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdense_1/kernel/Regularizer/mulhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b>gradient_tape/sequential/batch_normalization/batchnorm/mul/Mulhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28߸@?H?bdense/kernel/Regularizer/mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization/batchnorm/mul/Mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bBgradient_tape/sequential/batch_normalization_1/batchnorm/mul/Mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization_1/batchnorm/mul/Mulhu  ?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b4sequential/batch_normalization/AssignMovingAvg_1/mulhu  ?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28޵@?H?b2sequential/batch_normalization/AssignMovingAvg/mulhu  ?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b5gradient_tape/binary_crossentropy/logistic_loss/mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bBgradient_tape/sequential/batch_normalization/batchnorm/mul_2/Mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b@gradient_tape/sequential/batch_normalization/batchnorm/mul_2/Mulhu  ?B
~
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b;gradient_tape/sequential/batch_normalization_1/moments/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_2/Mul_1hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_2hu  ?B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bdiv_no_nan_2hu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b.sequential/batch_normalization/batchnorm/mul_2hu  ?B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28¯@?H?b7gradient_tape/binary_crossentropy/logistic_loss/mul/Mulhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bBgradient_tape/sequential/batch_normalization_1/batchnorm/mul_2/Mulhu  ?B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b6sequential/batch_normalization_1/AssignMovingAvg_1/mulhu  ?B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b0sequential/batch_normalization_1/batchnorm/mul_2hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?bmul_3hu  ?B
V
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?H?bGreaterEqualhu  ?B
?
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?b:gradient_tape/binary_crossentropy/logistic_loss/zeros_likehu  ?B
u
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28߫@?H?b3gradient_tape/binary_crossentropy/logistic_loss/Neghu  ?B
i
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b&gradient_tape/binary_crossentropy/Casthu  ?B
~
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b;gradient_tape/sequential/batch_normalization/moments/Cast_1hu  ?B
|
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b9gradient_tape/sequential/batch_normalization/moments/Casthu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28??@?H?b=gradient_tape/sequential/batch_normalization_1/moments/Cast_1hu  ?B
?
&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*?28ސ@?H?b<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1hu  ?B
D
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?hu  ?B