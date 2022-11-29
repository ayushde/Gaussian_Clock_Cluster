using GJRand
function alias_setup(probs)
    n=length(probs)
    Q=zeros(Float64,n)
    U=zeros(miniint,n)

    # Sort the probabilties that are smaller or larger than 1/n
    smalls=[]; larges=[]
    for (i,p) in enumerate(probs)
        Q[i]=n*p
        push!((Q[i]<1.0 ? smalls : larges),i)
    end

    while !isempty(smalls) && !isempty(larges)
        s=pop!(smalls); l=pop!(larges)
        U[s]=l
        Q[l]=Q[l]-(1.0-Q[s])
        push!((Q[l]<1.0 ? smalls : larges),l)
    end

    return U,Q
end
function alias_rand(U::Array{miniint,1},Q::Array{Float64,1},s::GJRandRNG)
    n=length(U)
    i=rand(s,1:n)
    if rand(s)<Q[i]
        return i
    else
        return U[i]
    end
end
