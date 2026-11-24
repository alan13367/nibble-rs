//! `nibble serve`: run the OpenAI-compatible HTTP server.

use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use nibble::Model;
use nibble_server::{ServerOptions, serve};

use crate::args::{Arg, Args, CliResult, MODEL_HELP, ModelOptions};
use crate::style::Style;

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

    let (model, load_time) = model_opts.load_quietly()?;
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

    let listener = TcpListener::bind((host.as_str(), port))
        .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let addr = listener.local_addr()?;
    let mut options = ServerOptions::new(model_id.clone());
    options.api_key = api_key;
    options.session = model_opts.session_options();

    // A throwaway session reports the context, threads and kernels the
    // server's own session will use; its buffers are not touched.
    let probe = model.session(options.session);
    let banner = Banner {
        model_id: &model_id,
        model: &model,
        file_bytes: model_opts
            .path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len()),
        load_time,
        n_ctx: probe.n_ctx(),
        threads: probe.threads(),
        kernel: probe.kernel(),
        addr,
        api_key: options.api_key.is_some(),
    };
    drop(probe);
    eprint!("{}", banner.render(Style::stderr()));
    serve(listener, model, options)?;
    Ok(())
}

/// What `nibble serve` prints once it is listening.
struct Banner<'a> {
    model_id: &'a str,
    model: &'a Model,
    file_bytes: Option<u64>,
    load_time: Duration,
    n_ctx: usize,
    threads: usize,
    kernel: &'static str,
    addr: SocketAddr,
    api_key: bool,
}

impl Banner<'_> {
    fn render(&self, s: Style) -> String {
        let c = self.model.config();
        let dot = s.dim(" · ");
        let row =
            |label: &str, value: String| format!("  {} {value}\n", s.dim(&format!("{label:<8}")));
        let mut out = format!(
            "\n  {} {}{dot}{}\n\n",
            s.title("nibble"),
            s.dim(env!("CARGO_PKG_VERSION")),
            s.dim("OpenAI-compatible server"),
        );

        let mut model = format!(
            "{}{dot}{}{dot}{}{dot}{} layers",
            s.bold(self.model_id),
            c.architecture.name(),
            c.weight_type,
            c.n_layer,
        );
        let how = if self.model.is_mapped() {
            "memory-mapped"
        } else {
            "read"
        };
        let size = self
            .file_bytes
            .map(|b| format!("{}, ", bytes(b)))
            .unwrap_or_default();
        let took = format!(
            "{size}{how} in {:.0} ms",
            self.load_time.as_secs_f64() * 1e3
        );
        model.push_str(&format!("  {}", s.dim(&took)));
        out.push_str(&row("model", model));

        let max = if self.n_ctx == c.n_ctx_train {
            s.dim("  (the model's maximum)")
        } else {
            s.dim(&format!("  (model maximum {})", c.n_ctx_train))
        };
        out.push_str(&row("context", format!("{} tokens{max}", self.n_ctx)));
        out.push_str(&row(
            "compute",
            format!("{} threads{dot}{} kernels", self.threads, self.kernel),
        ));
        let auth = if self.api_key {
            "API key required".to_owned()
        } else {
            format!("none {}", s.dim("(pass --api-key to require one)"))
        };
        out.push_str(&row("auth", auth));

        // An unspecified address accepts connections on every interface; the
        // local name is the one to open.
        let host = if self.addr.ip().is_unspecified() {
            format!("localhost:{}", self.addr.port())
        } else {
            self.addr.to_string()
        };
        let arrow = if s.enabled() { "➜" } else { "->" };
        out.push_str(&format!(
            "\n  {} {}\n\n",
            s.title(arrow),
            s.link(&format!("http://{host}/v1"))
        ));
        for (method, path) in [
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/completions"),
            ("GET", "/v1/models"),
            ("GET", "/health"),
        ] {
            out.push_str(&format!(
                "    {} {path}\n",
                s.keyword(&format!("{method:<4}"))
            ));
        }

        let mut warnings = Vec::new();
        if self.model.chat_template().is_none() {
            warnings.push(
                "the chat template is not supported, so /v1/chat/completions will fail".to_owned(),
            );
        }
        if !self.addr.ip().is_loopback() && !self.api_key {
            warnings.push(format!(
                "listening on {} without an API key: anyone who can reach it can use it",
                self.addr.ip()
            ));
        }
        if !warnings.is_empty() {
            out.push('\n');
        }
        for w in warnings {
            out.push_str(&format!("  {} {w}\n", s.warning("warning:")));
        }
        out.push_str(&format!("\n  {}\n\n", s.dim("Press Ctrl+C to stop.")));
        out
    }
}

/// A file size in MiB or GiB.
fn bytes(n: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let mib = n as f64 / MIB;
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

#[cfg(test)]
mod tests {
    use super::bytes;

    #[test]
    fn sizes_use_mib_below_a_gib() {
        assert_eq!(bytes(811_597_824), "774 MiB");
        assert_eq!(bytes(1_503_238_554), "1.4 GiB");
    }
}
