
�
�void gemv2N_kernel<int, int, float, float, float, 128, 32, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)@�*�28��@�H�0Xbsequential/dense_1/MatMulhu  HB
�
�void gemv2N_kernel<int, int, float, float, float, 128, 4, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)@�*�28�@�H�0Xbsequential/dense/MatMulhu  HB
�
�void gemv2T_kernel_val<int, int, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)(�*�28��@�H�(Xbsequential/dense_2/MatMulhu  �B
�
�void gemv2T_kernel_val<int, int, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)(�*�28��@�H� Xb.gradient_tape/sequential/dense_1/MatMul/MatMulhu  �B
�
�void gemmk1_kernel<float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28��@�H� Xb.gradient_tape/sequential/dense_2/MatMul/MatMulhu  �B
�
�void gemmk1_kernel<float, 256, 5, false, false, true, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28��@�H� b0gradient_tape/sequential/dense_1/MatMul/MatMul_1hu  �B
�
�void gemmk1_kernel<float, 256, 5, false, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28��@�H�b0gradient_tape/sequential/dense_2/MatMul/MatMul_1hu  �B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b RMSprop/RMSprop/update_4/truedivhu  �B
t
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b$mean_squared_error/SquaredDifferencehu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/addhu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *�28��@�H�b4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradhu  �B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/Squarehu  �B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b RMSprop/RMSprop/update_2/truedivhu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/Sqrthu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�bAssignAddVariableOp_2hu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/addhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!* 28��@�H�b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu �	B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H� bRMSprop/RMSprop/update_5/mulhu  �B
m
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b&mean_squared_error/weighted_loss/valuehu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/subhu  �B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b RMSprop/RMSprop/update_5/truedivhu  �B
M
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/subhu  �B
E
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bAbshu  �B
�
�void gemmk1_kernel<float, 256, 5, false, false, true, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28��@�H�Xb,gradient_tape/sequential/dense/MatMul/MatMulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�bAssignAddVariableOp_4hu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/add_1hu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28��@�H�bsequential/dense/BiasAddhu  �B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/Squarehu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/mulhu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/subhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/mul_1hu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/mul_1hu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�b)gradient_tape/sequential/dense_1/ReluGradhu  �B
`
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/truedivhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�bAssignAddVariableOphu  �B
h
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b&gradient_tape/mean_squared_error/mul_1hu  �B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b RMSprop/RMSprop/update_1/truedivhu  �B
X
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bsequential/dense/Reluhu  �B
f
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b$gradient_tape/mean_squared_error/subhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/mul_2hu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/add_1hu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/addhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/mul_2hu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/mulhu  �B
\
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/subhu  �B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b RMSprop/RMSprop/update_3/truedivhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�bAssignAddVariableOp_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�bAssignAddVariableOp_3hu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_5/mul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!* 28��@�H�b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu �	B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/mul_2hu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_2/Sqrthu  �B
S
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bdiv_no_nan_1hu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28��@�H�bsequential/dense_2/BiasAddhu  �B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_3/Squarehu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/mulhu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_3/mulhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_3/mul_2hu  �B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/Squarehu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/mul_1hu  �B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/Squarehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�b'gradient_tape/sequential/dense/ReluGradhu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/mul_2hu  �B
Q
%DivNoNan_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�b
div_no_nanhu  �B
Z
!Relu_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bsequential/dense_1/Reluhu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_3/add_1hu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28��@�H�b#RMSprop/RMSprop/AssignAddVariableOphu  �B
b
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/Squarehu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/mul_2hu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/add_1hu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/mulhu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_4/Sqrthu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_1/add_1hu  �B
^
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/addhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/mul_1hu  �B
^
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update/Sqrthu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�@�H�bsequential/dense_1/BiasAddhu  �B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_4/add_1hu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_3/addhu  �B
^
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_3/subhu  �B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/addhu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_3/mul_1hu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_3/Sqrthu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@�H�bRMSprop/RMSprop/update_1/Sqrthu  �B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�@�H�b
LogicalAndhu  �B