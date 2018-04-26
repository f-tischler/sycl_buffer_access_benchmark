file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)
 
foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})

   get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
   get_filename_component(ONE_BENCH_EXEC_DIR ${ONE_BENCH_CPP} DIRECTORY)

   set(LOG_NAME ${ONE_BENCH_EXEC_DIR}/lastrun/${ONE_BENCH_EXEC}.json)
   set(TARGET_NAME bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
   set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 14)

   target_include_directories(
		${TARGET_NAME} 
		PUBLIC
		${benchmark_INCLUDE_DIR}
		${BOOST_ALIGN_INCLUDE_DIR}
   )	

   target_link_libraries(${TARGET_NAME} 
	PRIVATE
		benchmark
		${CMAKE_THREAD_LIBS_INIT})

	add_sycl_to_target(
		${TARGET_NAME} 
		${CMAKE_CURRENT_BINARY_DIR}
		${ONE_BENCH_CPP}
	)

   add_test(NAME ${TARGET_NAME}
		    COMMAND ${TARGET_NAME} --benchmark_out=${LOG_NAME} 
								   --benchmark_out_format=json)
endforeach()