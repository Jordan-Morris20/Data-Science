
n
sgemm_32x32x32_NT_vec*28½ö
@’4HPXb0gradient_tape/sequential_2/dense_7/MatMul/MatMulhu  A
Y
sgemm_32x32x32_NN_vec*28č
@ß2H LXbsequential_2/dense_7/MatMulhu  A
j
sgemm_32x32x32_TN*28Ö
@Ą3HąLXb0gradient_tape/sequential_2/dense_6/MatMul/MatMulhu  A
n
sgemm_32x32x32_TN_vec*28“
@ą2HąIb2gradient_tape/sequential_2/dense_7/MatMul/MatMul_1hu  A
U
sgemm_32x32x32_NN*28¾
@ ,HNXbsequential_2/dense_6/MatMulhu  A
ó
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*28@ )Hą,b;sequential_2/dropout_2/dropout/random_uniform/RandomUniformhu  ČB
ż
«void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28 @ #H(b/gradient_tape/binary_crossentropy/DynamicStitchhu  ČB
Ž
void gemv2N_kernel<int, int, float, float, float, float, 128, 32, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)0*2 8”@ąHą,b2gradient_tape/sequential_2/dense_8/MatMul/MatMul_1hu  zB
ß
void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)3*28¾Ą@ąH'Xbsequential_2/dense_8/MatMulhu  aB

²void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!* 28¼@H(b6gradient_tape/sequential_2/dense_8/BiasAdd/BiasAddGradhu  ČB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28įŖ@ H $b$Adam/Adam/update_2/ResourceApplyAdamhu  ČB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¢@ąHĄ%b$Adam/Adam/update_5/ResourceApplyAdamhu  ČB
­
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *28Ą @ĄHąb6gradient_tape/sequential_2/dense_7/BiasAdd/BiasAddGradhu  ČB
ś
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¼@æH'b"Adam/Adam/update/ResourceApplyAdamhu  ČB
k
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*28@ HĄb'binary_crossentropy/logistic_loss/Log1phu  ČB
­
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *28@ H b6gradient_tape/sequential_2/dense_6/BiasAdd/BiasAddGradhu  ČB
÷	
æ	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@ąHbAdam/gradients/AddN_2hu  ČB
Ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ä@H b6gradient_tape/binary_crossentropy/weighted_loss/Tile_1hu  ČB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ą@H b$Adam/Adam/update_1/ResourceApplyAdamhu  ČB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28’ß@H b$Adam/Adam/update_3/ResourceApplyAdamhu  ČB
J
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28ĀŽ@ĄHąbAdam/Powhu  ČB

¹void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)*28Õ@ĄHĄXb0gradient_tape/sequential_2/dense_8/MatMul/MatMulhu  ČB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*2 8įÓ@’Hąbdense_7/kernel/Regularizer/Sumhu  ČB
ü
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ó@ Hßb$Adam/Adam/update_4/ResourceApplyAdamhu  ČB
Ŗ
ßvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*28Ņ@ąH b(ArithmeticOptimizer/AddOpsRewrite_AddN_1hu  HB

'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ē@ĄHąb:gradient_tape/binary_crossentropy/logistic_loss/Reciprocalhu  ČB
u
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28Ą¹@ąHb+sequential_2/dropout_2/dropout/GreaterEqualhu  ČB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28æ²@H b.gradient_tape/dense_7/kernel/Regularizer/Mul_1hu  ČB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28ą«@ Hbsequential_2/dense_6/BiasAddhu  ČB

Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28ż¦@Hąbdense_7/kernel/Regularizer/Sumhu  HB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28ą¦@ HĄbdense_6/kernel/Regularizer/Sumhu  ČB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28„@ HĄb2gradient_tape/sequential_2/dropout_2/dropout/Mul_1hu  ČB
Å
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28æ¢@ßHĄbAdam/gradients/AddNhu  ČB
d
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ą”@ĄHb"sequential_2/dropout_2/dropout/Mulhu  ČB
ģ
Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28””@ĄHįbSum_5hu  ČB
t
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*28’ @ Hb1gradient_tape/dense_7/kernel/Regularizer/Abs/Signhu  ČB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@ HĄb0gradient_tape/dense_7/kernel/Regularizer/Abs/mulhu  ČB
g
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*28@’HĄb%binary_crossentropy/logistic_loss/Exphu  ČB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*28£@ĄHąb!dense_7/kernel/Regularizer/Squarehu  ČB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28¢@ĄHįb dense_6/kernel/Regularizer/Sum_1hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@ßH bAssignAddVariableOp_5hu  ČB
G
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28¾@Hbaddhu  ČB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß@ĄHąb0gradient_tape/sequential_2/dropout_2/dropout/Mulhu  ČB
ļ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28’@æHb+gradient_tape/sequential_2/dense_6/ReluGradhu  ČB
ļ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ū@ßH b+gradient_tape/sequential_2/dense_7/ReluGradhu  ČB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ą@ĄHąbsequential_2/dense_7/BiasAddhu  ČB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*2 8æ@ĄH b dense_7/kernel/Regularizer/Sum_1hu  ČB
f
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@ĄH b$sequential_2/dropout_2/dropout/Mul_1hu  ČB
\
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*2@8@ąH’bsequential_2/dense_6/Reluhu  ČB
`
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ą@ąHbdense_7/kernel/Regularizer/Abshu  ČB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ć’@ąHbdense_7/kernel/Regularizer/mulhu  ČB
w
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28įõ@ąHĄb3gradient_tape/binary_crossentropy/logistic_loss/addhu  ČB
÷	
æ	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28žš@’HßbAdam/gradients/AddN_1hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ąš@ HąbAssignAddVariableOp_2hu  ČB
g
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28”š@ąH”b%binary_crossentropy/logistic_loss/subhu  ČB
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28 š@Hąb.binary_crossentropy/logistic_loss/GreaterEqualhu  ČB
L
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*28įī@ąH bGreaterhu  ČB

Łvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28’ģ@H bAssignAddVariableOp_6hu  ČB
E
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28ģ@ H bsubhu  ČB
e
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28ė@ąH b!binary_crossentropy/logistic_losshu  ČB
b
$Sigmoid_GPU_DT_FLOAT_DT_FLOAT_kernel*28ź@ąHąbsequential_2/dense_8/Sigmoidhu  ČB
ā
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ßē@Hb*binary_crossentropy/logistic_loss/Select_1hu  ČB
k
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28æē@ąHĄb)gradient_tape/binary_crossentropy/truedivhu  ČB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ē@ĄHĄbdiv_no_nan_1hu  ČB
L
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*28”ę@ HąbAdam/addhu  ČB

Łvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 ę@HĄbAdam/Adam/AssignAddVariableOphu  ČB
ģ
Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28Įć@ąHbSum_2hu  ČB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Ž@ßHĄbmul_4hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ü@ĄHĄbAssignAddVariableOphu  ČB
g
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ü@ĄH b%binary_crossentropy/logistic_loss/Neghu  ČB
ī
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ĄŚ@ĄHĄb6gradient_tape/binary_crossentropy/logistic_loss/Selecthu  ČB
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ś@ąHĄb
Adam/Pow_1hu  ČB
e
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*2@8 Ł@ąHąb#sequential_2/dropout_2/dropout/Casthu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28×@HbAssignAddVariableOp_3hu  ČB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28’Õ@ Hąb.gradient_tape/dense_6/kernel/Regularizer/Mul_1hu  ČB
Q
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28Ó@”HĄb
LogicalAndhu  ČB
G
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Ņ@HĄbsub_1hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ĮŠ@HąbAssignAddVariableOp_4hu  ČB
E
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28”Ļ@ßHąbMulhu  ČB
y
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28ĮĪ@ĄHĄb7gradient_tape/binary_crossentropy/logistic_loss/sub/Neghu  ČB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ćĶ@H b0gradient_tape/dense_6/kernel/Regularizer/Abs/mulhu  ČB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28įĶ@ Hąb%binary_crossentropy/weighted_loss/Sumhu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ķ@ HįbAssignAddVariableOp_1hu  ČB
\
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*2@8’Ė@ Hąbsequential_2/dense_7/Reluhu  ČB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ź@ßHąb3gradient_tape/binary_crossentropy/logistic_loss/mulhu  ČB
E
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28įÉ@ąHbAbshu  ČB
N
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28ĮĒ@æH bAdam/Cast_1hu  ČB
ģ
Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28”Ę@ H bSum_3hu  ČB
š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ŽÅ@Hąb8gradient_tape/binary_crossentropy/logistic_loss/Select_3hu  ČB
ģ
Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28Å@ HæbSum_4hu  ČB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*28įÄ@ Hb!dense_6/kernel/Regularizer/Squarehu  ČB
ą
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28żĮ@’H b(binary_crossentropy/logistic_loss/Selecthu  ČB

%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28įĮ@ĄHąb@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanhu  ČB
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28Į@ąHbadd_1hu  ČB

Åvoid tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28ĮĄ@HĄb dense_7/kernel/Regularizer/Sum_1hu  HB
t
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*28¾Ą@ĄHąb1gradient_tape/dense_6/kernel/Regularizer/Abs/Signhu  ČB
š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28 Ą@’HĄb8gradient_tape/binary_crossentropy/logistic_loss/Select_2hu  ČB
H
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*28Āæ@ĄHbCasthu  ČB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß¼@ĄHbdiv_no_nan_2hu  ČB
I
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ć¼@ąHĄbadd_2hu  ČB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28»@H”bmul_1hu  ČB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28¢ŗ@ßHĮb dense_7/kernel/Regularizer/mul_1hu  ČB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28”ŗ@ąHĄbmul_6hu  ČB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Į¹@ąHĄb dense_6/kernel/Regularizer/mul_1hu  ČB
n
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28¹@ĄHb'binary_crossentropy/weighted_loss/valuehu  ČB
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28æ·@ HbCast_1hu  ČB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28æ·@ßHbdense_6/kernel/Regularizer/mulhu  ČB
`
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28·@ Hbdense_6/kernel/Regularizer/Abshu  ČB
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28’µ@ĄH bdiv_no_nan_3hu  ČB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28’“@ HĄbsequential_2/dense_8/BiasAddhu  ČB
Q
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*28æ“@æHb
div_no_nanhu  ČB
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ą³@HĄb%binary_crossentropy/logistic_loss/mulhu  ČB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Æ@Hbmul_3hu  ČB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ß­@H bmul_2hu  ČB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 ­@H b7gradient_tape/binary_crossentropy/logistic_loss/mul/Mulhu  ČB
v
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28¬@ĄHąb3binary_crossentropy/weighted_loss/num_elements/Casthu  ČB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ą«@’HĄb5gradient_tape/binary_crossentropy/logistic_loss/mul_1hu  ČB
V
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28’Ŗ@HæbGreaterEqualhu  ČB

&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ŗ@æHb:gradient_tape/binary_crossentropy/logistic_loss/zeros_likehu  ČB
H
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*28 ©@ąHbCast_3hu  ČB
u
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*28 Ø@ßH b3gradient_tape/binary_crossentropy/logistic_loss/Neghu  ČB
i
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28@ H b&gradient_tape/binary_crossentropy/Casthu  ČB

&ZerosLike_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ą@ĄHb<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1hu  ČB