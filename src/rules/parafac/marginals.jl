struct CPTensorMarginal{T,N,M,C}
    cp::CPTensor{T,N}
    incoming_msg::NTuple{N,Vector{T}}
end

