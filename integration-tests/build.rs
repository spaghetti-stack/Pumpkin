use std::error::Error;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

fn main() {
    if let Err(e) = run() {
        fail(&e.to_string());
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    compile_c_binary(&["src/drat-trim.c"], "drat-trim")?;
    compile_c_binary(&["src/precochk.c"], "precochk")?;
    compile_c_binary(&["src/maxsat-checker.cc", "src/Wcnf.cc"], "maxsat-checker")?;

    println!("cargo:rerun-if-changed=src/*.c");
    println!("cargo:rerun-if-changed=src/*.h");
    println!("cargo:rerun-if-changed=src/*.cc");
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}

fn compile_c_binary<Source: AsRef<Path>>(
    sources: &[Source],
    output_stem: &str,
) -> Result<(), Box<dyn Error>> {
    let mut build = cc::Build::new();
    build.opt_level(2);

    if sources
        .iter()
        .any(|source| source.as_ref().extension() == Some(OsStr::new("cc")))
    {
        build.cpp(true);
        build.std("c++17");
    }

    let compiler = build.try_get_compiler()?;
    let mut cmd = compiler.to_command();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let output_dir = Path::new(&out_dir);

    add_output_file(&mut cmd, compiler.is_like_msvc(), output_dir, output_stem);

    for source in sources {
        cmd.arg(source.as_ref());
    }

    let status_code = cmd.status()?;

    if status_code.success() {
        Ok(())
    } else {
        Err("Failed to compile binary".into())
    }
}

fn add_output_file<P: AsRef<Path>>(
    cmd: &mut Command,
    is_msvc: bool,
    output_dir: P,
    output_stem: &str,
) {
    let output_dir = output_dir.as_ref();
    if is_msvc {
        let exe_name = format!("{output_stem}.exe");

        // The path to the object file.
        cmd.arg(format!("/Fo:{}/", output_dir.to_string_lossy()));

        // The path to the executable.
        cmd.arg(format!("/Fe:{}/{exe_name}", output_dir.to_string_lossy()));

        cmd.arg("/std:c++17");
    } else {
        let output_file = output_dir.join(output_stem);
        cmd.arg("-o").arg(output_file);
    }
}

fn fail(s: &str) -> ! {
    eprintln!("\n\nerror occurred: {}\n\n", s);
    std::process::exit(1);
}
