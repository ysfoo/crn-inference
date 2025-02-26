using BenchmarkTools

data = zeros(10,20)
comp = [vec(1:10) for _ in 1:20]

function func1(x, y)
    return sum(abs2.(x .- reduce(hcat, y)))
end
    
function func2(x, y)
    loss = 0.0
    for idx in CartesianIndices(x)
        i, j = Tuple(idx)
        @inbounds loss += abs2(x[i,j] - y[j][i])
    end
    return loss
end

@btime func1($data, $comp)
@btime func2($data, $comp)