$	???y??????ժ???`\:???!?>??V??$	ހ?=?@]???ߛ
@ 7o?}?@!??Y`1@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC? :v???l$	?P??1??9???A`"ĕ??I?U?6????YY??????r	train 197"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCN?g\???R?.?????1
?]?V??AdX??G??I?!???Y?Ŧ?B??r	train 198"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????z??E??S???1??????AB?۽?'??Ih^?????Y???????r	train 199"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?SV??D???G?C????1?%????A????????I???}?A??YKXc'??r	train 200"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCE?e?????З??\4??1b?o???A@x?=\??I&??)d??Y?????پ?r	train 201"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC|~!<?????ܴ??1??	?_???A[%X???I?^?sa$??Y???cꮰ?r	train 202"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?ao$??&qVDM???1jM??St??A????I??????Y~t??gy??r	train 203"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC	_?2??z???GnM?-??1?+?PO??A-?\o????I??????YʋL?????r	train 204"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC
<3?p?!???@?شR??1\;Qi??AF?-t%??I?|zlˀ??Y"6X8I???r	train 205"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?
E???$???x??1???d#??A?h?"???IN??????YlMK????r	train 206"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?[[?????s????1?_vO??A??V`????IYQ?i???Y???Z(??r	train 207"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?`\:???m????19?@d?&??A6:??8??IJ?%r???Y?@?S????r	train 208"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??C???????lY??1%]3?f???A ???-???I?9???1??Yxρ???r	train 209"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC0?G?????WV?????1?o?h????A?.????IwJ???Yf??Os??r	train 210"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCv?;O<'??Ҭl????1??e???A??	h"??I?u?+.??Y.??:???r	train 211"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCPqx?????ȓ?k&??1??=?#??AmT?YO??IOqN???Ygc%?YI??r	train 212"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?>??V??-@?j???1???0??A4??O??I???
??Y??o???r	train 213"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????e??_?BF???1??d?<??AmV}??b??I??8ӄm??Y?8h???r	train 214"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Co??????Q????1?zO崧??Aτ&?%???I<-?p?'??Yk??qQ-??r	train 215*	??ʡE??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?46<??!
;$B@)I+??????1?j?~??@@:Preprocessing2T
Iterator::Root::ParallelMapV2???ฌ??! ?<Ps3@)???ฌ??1 ?<Ps3@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSliceo??\????!y߱I?{0@)o??\????1y߱I?{0@:Preprocessing2E
Iterator::Root?$#gaO??!??S?4&?@)??8Q???1?-?bf'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip'?y?3M??!???r6Q@)g*?#????1????? @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?????B??!?I??#8@)??,?Ů?1-??ב?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?]?V$??!r?ڲ8@)?]?V$??1r?ڲ8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9ƫ?I @I?[?;?S@Q1?%ڹ+@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Ӆg$E^??w??????E??S???!-@?j???	!       "$	?W?*?:???R^S????=?#??!?%????*	!       2$	>?.'???$?\???6:??8??!mV}??b??:$	?Ϧ????Mj?騪?J?%r???!OqN???B	!       J$	??y?M???fc3???ʋL?????!~t??gy??R	!       Z$	??y?M???fc3???ʋL?????!~t??gy??b	!       JGPUYƫ?I @b q?[?;?S@y1?%ڹ+@