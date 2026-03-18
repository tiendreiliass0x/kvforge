fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=kernels/");

        let mut build = cc::Build::new();
        build.cuda(true);
        build.flag("-gencode=arch=compute_80,code=sm_80");
        build.flag("-gencode=arch=compute_89,code=sm_89");
        build.flag("-gencode=arch=compute_90,code=sm_90");

        build.file("kernels/fused_project_quantize.cu");
        build.file("kernels/fused_dequant_attention.cu");
        build.file("kernels/entropy_encode.cu");
        build.file("kernels/entropy_decode_dequant.cu");

        build.compile("kvforge_kernels");
    }
}
