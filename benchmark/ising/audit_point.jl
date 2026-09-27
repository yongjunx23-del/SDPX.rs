# Independent audit of a native Rust CLI point. No SDPX Julia package is loaded.
length(ARGS)==5 || error("usage: audit_point.jl INPUT RAW_JSON OUTPUT_JSON BITS REFERENCE_AUDIT")
const input_dir, raw_path, output_path, bits_text, reference_path = ARGS
const bits = parse(Int,bits_text)
bits in (512,768) || error("unsupported Ising audit precision")
const tolerance_text = "1e-30"
include("audit_helpers.jl")
BLAS.set_num_threads(1)

setprecision(BigFloat,bits) do
    raw=JSON.parsefile(raw_path)
    raw["precision_bits"]==bits || error("point precision mismatch")
    reference=JSON.parsefile(reference_path)
    reference["accepted"]===true || error("reference was not accepted")
    sampled=read_sampled_sdp(input_dir;T=BigFloat,bits)
    reference_hash=get(reference,"input_sha256",get(get(reference,"mapping",Dict()),"input_sha256",nothing))
    reference_hash==sampled.input_sha256 || error("reference input hash mismatch")
    conic=compile_sampled_primal(sampled)
    x,s,z=(parse.(BigFloat,raw[k]) for k in ("x","s","z"))
    a=audit(conic,x,s,z)
    a["mapping_relative"]=mapping_gate_value(a["mapping"])
    ref_objective=parse(BigFloat,reference["objective"])
    a["reference_objective_agreement"]=abs(a["objective"]-ref_objective)/
        max(one(BigFloat),abs(a["objective"]),abs(ref_objective))
    a["optimal"]=raw["status"]=="Solved"
    fields=("primal","dual","gap","primal_psd","dual_psd","mapping_relative","reference_objective_agreement")
    a["accepted"]=a["optimal"] && isfinite(ref_objective) &&
        all(isfinite(a[k]) && a[k]<=parse(BigFloat,tolerance_text) for k in fields)
    a["input_sha256"]=sampled.input_sha256
    a["precision_bits"]=bits
    a["point_sha256"]=bytes2hex(sha256(read(raw_path)))
    a["reference_sha256"]=bytes2hex(sha256(read(reference_path)))
    open(output_path,"w") do io; JSON.print(io,stringify(a),2);end
    println("accepted=",a["accepted"]," status=",raw["status"]," iterations=",raw["iterations"])
    a["accepted"] || error("native point failed original-coordinate Ising audit")
end
