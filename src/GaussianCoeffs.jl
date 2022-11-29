using FFTW

function gaussian_coeffs(N::Int64)
    c=1
    a=zeros(N)
    for (i,n) in enumerate(a)
        for j in -5:5
            a[i]=a[i]+exp(-((pi/(c*N))*((i-1+N*j)^2)))
        end
    end
    return a
end
