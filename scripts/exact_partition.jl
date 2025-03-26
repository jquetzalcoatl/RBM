

function generate_binary_states(N::Int)
    return [digits(i, base=2, pad=N) for i in 0:(2^N - 1)]
end


function partition_function(J)
    Jw = cpu(J.w)
    Jb = cpu(J.b)
    Ja = cpu(J.a)
    vs = generate_binary_states(size(J.a,1))
    hs = generate_binary_states(size(J.b,1));
    sum([exp(v' * Ja + h' * Jb + v' * Jw * h) for v in vs, h in hs])
end

function partition_function(N::Int, J)
    Jw = cpu(J.w)
    Jb = cpu(J.b)
    Ja = cpu(J.a)
    vs = generate_binary_states(N)
    hs = generate_binary_states(N);
    sum([exp(v' * Ja + h' * Jb + v' * Jw * h) for v in vs, h in hs])
end


