
f
sgemm_32x32x32_TN*28¢¨@à5HRXb,gradient_tape/sequential/dense/MatMul/MatMulhu  A
l
sgemm_32x32x32_NT_vec*28¤ @À5H OXb.gradient_tape/sequential/dense_1/MatMul/MatMulhu  A
Q
sgemm_32x32x32_NN*28Ü@-HTXbsequential/dense/MatMulhu  A
l
sgemm_32x32x32_TN_vec*28Ò@3HáJb0gradient_tape/sequential/dense_1/MatMul/MatMul_1hu  A
W
sgemm_32x32x32_NN_vec*28Ï@à4HPXbsequential/dense_1/MatMulhu  A
Ü
void gemv2N_kernel<int, int, float, float, float, float, 128, 32, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)0*2 8¡@ "HÀ.b0gradient_tape/sequential/dense_2/MatMul/MatMul_1hu  zB
ï
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*28¡@ )H,b7sequential/dropout/dropout/random_uniform/RandomUniformhu  ÈB
ý
«void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28@à#H 'b/gradient_tape/binary_crossentropy/DynamicStitchhu  ÈB
Ý
void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)3*28£ú@àH 'Xbsequential/dense_2/MatMulhu  aB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28£ò@ Hà5b$Adam/Adam/update_4/ResourceApplyAdamhu  ÈB

²void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!* 28½î@¿HÀ+b4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradhu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Âá@ HÀ*b$Adam/Adam/update_9/ResourceApplyAdamhu  ÈB
Ð
void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28áæ@Hb+sequential/batch_normalization/moments/meanhu  ÈB
|
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28ãÜ@ HÀb:gradient_tape/sequential/batch_normalization_1/moments/subhu  ÈB

.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Ú@¿Hb8sequential/batch_normalization/moments/SquaredDifferencehu  ÈB
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢×@ÀH b.sequential/batch_normalization/batchnorm/add_1hu  ÈB
Ò
void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28àÑ@ÀHb-sequential/batch_normalization_1/moments/meanhu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ñ@ H %b$Adam/Adam/update_5/ResourceApplyAdamhu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÁÏ@ Hàb.sequential/batch_normalization/batchnorm/mul_1hu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¿Ï@ H bBgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Mulhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÄÈ@ Hb@gradient_tape/sequential/batch_normalization/batchnorm/mul_1/Mulhu  ÈB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÀÇ@H b0sequential/batch_normalization_1/batchnorm/mul_1hu  ÈB
z
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÿÆ@àHÀb8gradient_tape/sequential/batch_normalization/moments/subhu  ÈB
t
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28¦Æ@ Hb0sequential/batch_normalization_1/batchnorm/add_1hu  ÈB

.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Ä@ÀHÀb:sequential/batch_normalization_1/moments/SquaredDifferencehu  ÈB
«
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *28¾À@H $b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu  ÈB
Ô
void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28Ã½@àH b/sequential/batch_normalization/moments/variancehu  ÈB
Ö
void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28»@àHáb1sequential/batch_normalization_1/moments/variancehu  ÈB
è
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Á¯@ÀHàbBgradient_tape/sequential/batch_normalization_1/moments/BroadcastTohu  ÈB
ú
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28à¥@HÀb"Adam/Adam/update/ResourceApplyAdamhu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¾¥@ÀHà%b$Adam/Adam/update_1/ResourceApplyAdamhu  ÈB
k
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @Hb'binary_crossentropy/logistic_loss/Log1phu  ÈB
Ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28¢@ Hb6gradient_tape/binary_crossentropy/weighted_loss/Tile_1hu  ÈB
©
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *28@àHàb2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu  ÈB
æ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28á@ÀHÁb@gradient_tape/sequential/batch_normalization/moments/BroadcastTohu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28à@ÀHà%b$Adam/Adam/update_6/ResourceApplyAdamhu  ÈB
÷	
¿	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@ßHbAdam/gradients/AddN_2hu  ÈB

Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28¡@Hb+sequential/batch_normalization/moments/meanhu  ÈB
¬
Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28¡ÿ@H bDgradient_tape/sequential/batch_normalization_1/batchnorm/add_1/Sum_1hu  ÈB
J
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28Áþ@HàbAdam/Powhu  ÈB
è
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Áþ@ÀHàbBgradient_tape/sequential/batch_normalization/moments/BroadcastTo_1hu  ÈB

¹void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)*28þ@HÀXb.gradient_tape/sequential/dense_2/MatMul/MatMulhu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ø@àH&b$Adam/Adam/update_7/ResourceApplyAdamhu  ÈB
ê
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28â÷@ÀHbDgradient_tape/sequential/batch_normalization_1/moments/BroadcastTo_1hu  ÈB
ª
ßvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*28À÷@ Hb(ArithmeticOptimizer/AddOpsRewrite_AddN_1hu  HB
÷	
¿	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Âö@¡Há bAdam/gradients/AddN_6hu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Áó@¿Hàb$Adam/Adam/update_2/ResourceApplyAdamhu  ÈB
«
Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28Àò@àH bDgradient_tape/sequential/batch_normalization_1/batchnorm/add_1/Sum_1hu  ÈB

Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*2 8Àñ@àHá bdense_1/kernel/Regularizer/Sumhu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¢ì@HÁb$Adam/Adam/update_3/ResourceApplyAdamhu  ÈB
©
Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28Áé@ßHbBgradient_tape/sequential/batch_normalization/batchnorm/add_1/Sum_1hu  ÈB
ª
Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28Áä@àH bBgradient_tape/sequential/batch_normalization/batchnorm/add_1/Sum_1hu  ÈB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡ä@H b>gradient_tape/sequential/batch_normalization/moments/truediv_1hu  ÈB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28ã@H b@gradient_tape/sequential/batch_normalization_1/moments/truediv_1hu  ÈB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28áØ@ÀHÀb$Adam/Adam/update_8/ResourceApplyAdamhu  ÈB
t
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡Õ@àH#b1gradient_tape/dense_1/kernel/Regularizer/Abs/Signhu  ÈB

'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*28áÔ@ÀHb:gradient_tape/binary_crossentropy/logistic_loss/Reciprocalhu  ÈB
«
Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28Ó@àHbDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Sum_1hu  ÈB
©
Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  28Ò@àHàbBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Sum_1hu  ÈB

Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28Ë@Hb-sequential/batch_normalization_1/moments/meanhu  ÈB
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28Ê@àHàb'sequential/dropout/dropout/GreaterEqualhu  ÈB
÷	
¿	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ßÈ@ H bAdam/gradients/AddN_4hu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Å@àH b.gradient_tape/dense_1/kernel/Regularizer/Mul_1hu  ÈB
ì
Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28¿¶@ HÀbSum_5hu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28³@ÀHÀbAssignAddVariableOp_5hu  ÈB
g
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*28±@H b%binary_crossentropy/logistic_loss/Exphu  ÈB
Å
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28á°@àHbAdam/gradients/AddNhu  ÈB
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28°@¡HÀb:gradient_tape/sequential/batch_normalization_1/moments/Mulhu  ÈB
¬
Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28á­@àHÁbDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Sum_1hu  ÈB

Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28¢­@ÀH bdense/kernel/Regularizer/Sumhu  ÈB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28ÿ¬@ÀHbsequential/dense/BiasAddhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß¬@àHàbDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_1/Mul_1hu  ÈB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*28 «@ÀHb!dense_1/kernel/Regularizer/Squarehu  ÈB
À
Ûvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¢ª@ÀH bBgradient_tape/sequential/batch_normalization_1/batchnorm/RsqrtGradhu  ÈB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28§@H b>gradient_tape/sequential/batch_normalization_1/moments/truedivhu  ÈB
~
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28à¦@HÀb<gradient_tape/sequential/batch_normalization/moments/truedivhu  ÈB

Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28ß¦@àHÀb/sequential/batch_normalization/moments/variancehu  ÈB

Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28¥@àHb1sequential/batch_normalization_1/moments/variancehu  ÈB
ª
Ãvoid tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)!*  28¥@àH bBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Sum_1hu  ÈB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡¤@àHÿb0gradient_tape/dense_1/kernel/Regularizer/Abs/mulhu  ÈB

Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28Á£@ H bdense/kernel/Regularizer/Sum_1hu  ÈB

Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28À£@H bdense_1/kernel/Regularizer/Sumhu  HB
§
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28££@àH bAdam/gradients/AddN_1hu  ÈB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28ã @àH bsequential/dense_1/BiasAddhu  ÈB
G
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢@Hbaddhu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ß@ÀH bAssignAddVariableOp_2hu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @àHàb.gradient_tape/sequential/dropout/dropout/Mul_1hu  ÈB

Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*2 8ÿ@ÀHÀb dense_1/kernel/Regularizer/Sum_1hu  ÈB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb,gradient_tape/sequential/dropout/dropout/Mulhu  ÈB
í
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28@àHb)gradient_tape/sequential/dense_1/ReluGradhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @àHbBgradient_tape/sequential/batch_normalization/batchnorm/mul_1/Mul_1hu  ÈB
`
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡@ Hábdense_1/kernel/Regularizer/Abshu  ÈB
Â
ñvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@Hàb.sequential/batch_normalization/AssignMovingAvghu  ÈB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß@ Hßbsequential/dropout/dropout/Mulhu  ÈB
ë
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28@ÀHàb'gradient_tape/sequential/dense/ReluGradhu  ÈB
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¾@àH b8gradient_tape/sequential/batch_normalization/moments/Mulhu  ÈB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28á@àH¡b sequential/dropout/dropout/Mul_1hu  ÈB
ì
Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28Ã@ÀHbSum_2hu  ÈB

Ùvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@ H bAssignAddVariableOp_6hu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¤@ÀHÀbAssignAddVariableOphu  ÈB
X
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*2@8ã@àHbsequential/dense/Reluhu  ÈB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28à@àHÀb<gradient_tape/sequential/batch_normalization_1/moments/mul_1hu  ÈB
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@ÀHÀb:gradient_tape/sequential/batch_normalization/moments/mul_1hu  ÈB
t
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Háb2sequential/batch_normalization/AssignMovingAvg/subhu  ÈB
L
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*28Â@áH bGreaterhu  ÈB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ Hábdense_1/kernel/Regularizer/mulhu  ÈB

Ùvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28àþ@ H bAdam/Adam/AssignAddVariableOphu  ÈB
¾
Ûvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 þ@ H b@gradient_tape/sequential/batch_normalization/batchnorm/RsqrtGradhu  ÈB
`
$Sigmoid_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡ý@H bsequential/dense_2/Sigmoidhu  ÈB
E
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28ý@àHàbsubhu  ÈB
÷	
¿	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Âú@ HÀbAdam/gradients/AddN_5hu  ÈB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ú@àHÀbmul_4hu  ÈB
k
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28Áø@àH b)gradient_tape/binary_crossentropy/truedivhu  ÈB
p
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28ø@àH b,sequential/batch_normalization/batchnorm/addhu  ÈB
§
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Áö@àHàbAdam/gradients/AddN_3hu  ÈB
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28Áó@HÀb.binary_crossentropy/logistic_loss/GreaterEqualhu  ÈB
e
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28Àð@HÀb!binary_crossentropy/logistic_losshu  ÈB
w
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢ð@àHb3gradient_tape/binary_crossentropy/logistic_loss/addhu  ÈB
r
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ï@H b.sequential/batch_normalization/batchnorm/Rsqrthu  ÈB
î
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28áî@HÀb6gradient_tape/binary_crossentropy/logistic_loss/Selecthu  ÈB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28âí@ÀHÀbdiv_no_nan_1hu  ÈB
â
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Âê@ HÀb*binary_crossentropy/logistic_loss/Select_1hu  ÈB
g
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28¿ê@¿HÀb%binary_crossentropy/logistic_loss/Neghu  ÈB
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ê@àHÀb.sequential/batch_normalization_1/batchnorm/addhu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¡ç@ HbAssignAddVariableOp_4hu  ÈB
Z
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*2@8¿æ@ÀHbsequential/dense_1/Reluhu  ÈB
G
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28å@ÀH bsub_1hu  ÈB
Ä
ñvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ââ@ H b0sequential/batch_normalization_1/AssignMovingAvghu  ÈB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Âà@ Hb3gradient_tape/binary_crossentropy/logistic_loss/mulhu  ÈB
a
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*2@8¢à@HÀbsequential/dropout/dropout/Casthu  ÈB
E
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡Ý@àHàbMulhu  ÈB
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28âÛ@ÀHÀb4sequential/batch_normalization_1/AssignMovingAvg/subhu  ÈB
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢Ù@àHb
Adam/Pow_1hu  ÈB
E
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ù@ÀHbAbshu  ÈB
y
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28âØ@àHÀb7gradient_tape/binary_crossentropy/logistic_loss/sub/Neghu  ÈB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡Ø@¡Hàb,gradient_tape/dense/kernel/Regularizer/Mul_1hu  ÈB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡×@ÀHÀb>gradient_tape/sequential/batch_normalization/batchnorm/sub/Neghu  ÈB
L
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*28ÁÕ@ÀHàbAdam/addhu  ÈB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28áÔ@HÀb2sequential/batch_normalization/AssignMovingAvg/mulhu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 Ô@ HÀbAssignAddVariableOp_3hu  ÈB

ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28àÓ@ HàbAssignAddVariableOp_1hu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÞÒ@¿H¿b.gradient_tape/dense/kernel/Regularizer/Abs/mulhu  ÈB
Æ
ñvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ñ@ Háb2sequential/batch_normalization_1/AssignMovingAvg_1hu  ÈB
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ñ@àHàb4sequential/batch_normalization/AssignMovingAvg_1/subhu  ÈB
Q
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28ÃÏ@ Hb
LogicalAndhu  ÈB
Ä
ñvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 Î@ÀHÀb0sequential/batch_normalization/AssignMovingAvg_1hu  ÈB
H
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*28½Í@ÀHbCasthu  ÈB

%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28£Ì@ÀH b@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanhu  ÈB
ð
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28 Ì@Hàb8gradient_tape/binary_crossentropy/logistic_loss/Select_2hu  ÈB
r
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ì@àHàb/gradient_tape/dense/kernel/Regularizer/Abs/Signhu  ÈB
ì
Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28Ë@ HàbSum_4hu  ÈB
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28äÉ@ÁH bCast_1hu  ÈB
t
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*28âÉ@H¡b0sequential/batch_normalization_1/batchnorm/Rsqrthu  ÈB

Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28ÀÉ@ Hàb%binary_crossentropy/weighted_loss/Sumhu  ÈB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28É@ H¡b@gradient_tape/sequential/batch_normalization_1/batchnorm/sub/Neghu  ÈB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28àÈ@ÀHÀbsequential/dense_2/BiasAddhu  ÈB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28È@Hb,sequential/batch_normalization/batchnorm/mulhu  ÈB
N
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28 Ç@ HàbAdam/Cast_1hu  ÈB
Q
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÿÆ@¿Hb
div_no_nanhu  ÈB
ð
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ÁÆ@H b8gradient_tape/binary_crossentropy/logistic_loss/Select_3hu  ÈB
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÃÅ@àH badd_2hu  ÈB
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÁÅ@ H b.sequential/batch_normalization_1/batchnorm/subhu  ÈB
x
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28Å@ Hàb6sequential/batch_normalization_1/AssignMovingAvg_1/subhu  ÈB
ì
Âvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28ÀÄ@HàbSum_3hu  ÈB
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ä@¿HÀb%binary_crossentropy/logistic_loss/mulhu  ÈB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28âÃ@ÀHábdiv_no_nan_2hu  ÈB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28áÂ@ HÀb4sequential/batch_normalization_1/AssignMovingAvg/mulhu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¾Â@ HÀb.sequential/batch_normalization_1/batchnorm/mulhu  ÈB
à
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ÿÁ@Hàb(binary_crossentropy/logistic_loss/Selecthu  ÈB
n
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ¿@ Hb,sequential/batch_normalization/batchnorm/subhu  ÈB
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*28¿@ßHÀbdense/kernel/Regularizer/Squarehu  ÈB

Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28¡¾@HÁb dense_1/kernel/Regularizer/Sum_1hu  HB
n
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28À½@ÀHb'binary_crossentropy/weighted_loss/valuehu  ÈB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28À½@ÀHÀbdiv_no_nan_3hu  ÈB
g
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28¼@ÀH b%binary_crossentropy/logistic_loss/subhu  ÈB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28à»@Hbmul_2hu  ÈB
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28»@ßHàbadd_1hu  ÈB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28»@àHbmul_6hu  ÈB
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÿº@ßH bdense/kernel/Regularizer/mulhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ÿ¹@HÀb@gradient_tape/sequential/batch_normalization_1/batchnorm/mul/Mulhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28á¹@HàbDgradient_tape/sequential/batch_normalization_1/batchnorm/mul_2/Mul_1hu  ÈB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ¹@àHàb dense_1/kernel/Regularizer/mul_1hu  ÈB
^
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28â¸@ Hbdense/kernel/Regularizer/Abshu  ÈB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28á·@àHbdense/kernel/Regularizer/mul_1hu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28à·@H b@gradient_tape/sequential/batch_normalization/batchnorm/mul/Mul_1hu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡·@H bBgradient_tape/sequential/batch_normalization_1/batchnorm/mul/Mul_1hu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ·@HÀbBgradient_tape/sequential/batch_normalization_1/batchnorm/mul_2/Mulhu  ÈB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28â¶@ÀHÀbmul_1hu  ÈB
~
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28à¶@ÀHb;gradient_tape/sequential/batch_normalization_1/moments/Casthu  ÈB

&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*28Á¶@ H b:gradient_tape/binary_crossentropy/logistic_loss/zeros_likehu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¡¶@HÀb>gradient_tape/sequential/batch_normalization/batchnorm/mul/Mulhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Áµ@ HÀbBgradient_tape/sequential/batch_normalization/batchnorm/mul_2/Mul_1hu  ÈB
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*28Â´@ HÀbCast_3hu  ÈB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢³@ Hb7gradient_tape/binary_crossentropy/logistic_loss/mul/Mulhu  ÈB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28²@Hàb0sequential/batch_normalization_1/batchnorm/mul_2hu  ÈB
v
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28â±@ÀHÀb3binary_crossentropy/weighted_loss/num_elements/Casthu  ÈB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28À¯@HÀb.sequential/batch_normalization/batchnorm/mul_2hu  ÈB

!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28¯@ Hb=gradient_tape/sequential/batch_normalization_1/moments/Cast_1hu  ÈB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß®@àH b4sequential/batch_normalization/AssignMovingAvg_1/mulhu  ÈB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28®@Hàb@gradient_tape/sequential/batch_normalization/batchnorm/mul_2/Mulhu  ÈB
u
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28á­@àHÀb3gradient_tape/binary_crossentropy/logistic_loss/Neghu  ÈB
i
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28¡¬@ Hàb&gradient_tape/binary_crossentropy/Casthu  ÈB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¬@H b5gradient_tape/binary_crossentropy/logistic_loss/mul_1hu  ÈB
V
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28¬@HbGreaterEqualhu  ÈB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28à«@Hàbmul_3hu  ÈB
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28á©@Hb6sequential/batch_normalization_1/AssignMovingAvg_1/mulhu  ÈB
|
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28á¨@ÀHb9gradient_tape/sequential/batch_normalization/moments/Casthu  ÈB
~
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28 §@ÀH b;gradient_tape/sequential/batch_normalization/moments/Cast_1hu  ÈB

&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*28À@¿HÀb<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1hu  ÈB