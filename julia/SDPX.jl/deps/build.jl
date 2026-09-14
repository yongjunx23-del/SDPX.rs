# Build from the new Rust workspace only; no stable Julia solver dependency.
root=normpath(joinpath(@__DIR__,"..","..",".."))
cargo=get(ENV,"CARGO",something(Sys.which("cargo"),""))
if isempty(cargo)
    cargo_home=get(ENV,"CARGO_HOME",joinpath(homedir(),".cargo"))
    cargo=joinpath(cargo_home,"bin",Sys.iswindows() ? "cargo.exe" : "cargo")
end
isfile(cargo) || error("Cargo not found; install Rust and set CARGO or PATH")
features=Sys.isapple() ? "sdp-accelerate,faer-sparse" : "sdp-openblas,faer-sparse"
cd(root) do
    run(`$cargo build --locked --release -p sdpx-ffi --features $features`)
end
