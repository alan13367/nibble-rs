//! `nibble serve`: run the OpenAI-compatible HTTP server.

use std::net::TcpListener;

use nibble_server::{ServerOptions, serve};

use crate::args::{Arg, Args, CliResult, MODEL_HELP, ModelOptions};

pub(crate) fn help() -> String {
    format!(
        "Usage: nibble serve -m <model.gguf> [options]\n\n\
         Serves the OpenAI API: POST /v1/chat/completions, POST /v1/completions,\n\
         GET /v1/models, GET /health.\n\n\
         Server:\n      \
         --host ADDR          Address to listen on (default 127.0.0.1)\n      \
         --port N             Port (default 8080)\n      \
         --api-key KEY        Require `Authorization: Bearer KEY` (or set NIBBLE_API_KEY)\n      \
         --model-id NAME      Model name reported to clients (default: the model's name)\n\n\
         Model:\n{MODEL_HELP}"
    )
}

pub(crate) fn run(mut args: Args) -> CliResult<()> {
    let mut model_opts = ModelOptions::new();
    let (mut host, mut port) = ("127.0.0.1".to_owned(), 8080u16);
    let mut api_key = std::env::var("NIBBLE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let mut model_id = None;
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f) if model_opts.parse(&f, &mut args)? => {}
            Arg::Flag(f) => match f.as_str() {
                "--host" => host = args.string(&f)?,
                "--port" => port = args.parse(&f)?,
                "--api-key" => api_key = Some(args.string(&f)?),
                "--model-id" => model_id = Some(args.string(&f)?),
                "-h" | "--help" => {
                    print!("{}", help());
                    return Ok(());
                }
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble serve --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) => {
                return Err(format!("unexpected argument `{}`", p.to_string_lossy()).into());
            }
        }
    }

    let model = model_opts.load()?;
    let model_id = model_id
        .or_else(|| model.config().name.clone())
        .or_else(|| {
            model_opts
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "nibble".to_owned());
    if model.chat_template().is_none() {
        eprintln!(
            "nibble: warning: the chat template is not supported, so /v1/chat/completions will fail"
        );
    }

    let listener = TcpListener::bind((host.as_str(), port))
        .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let addr = listener.local_addr()?;
    let mut options = ServerOptions::new(model_id.clone());
    options.api_key = api_key;
    options.session = model_opts.session_options();
    eprintln!(
        "nibble: serving `{model_id}` on http://{addr}/v1{}",
        if options.api_key.is_some() {
            " (API key required)"
        } else {
            ""
        }
    );
    serve(listener, model, options)?;
    Ok(())
}
