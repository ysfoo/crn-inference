Threads.@threads for (a, b) in collect(Iterators.product(1:5, 1:10))
	println("$(Threads.threadid()) $(10*a+b)")
end

display(Threads.nthreads())