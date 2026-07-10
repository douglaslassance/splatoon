import Foundation
import Network

/// A tiny localhost HTTP server that serves registered files with permissive
/// CORS, so a web page (SuperSplat) can fetch a *local* splat via
/// `?load=http://localhost:PORT/name.ply` — browsers can't open a local file
/// directly. The app isn't sandboxed, so binding a loopback port is unrestricted.
@MainActor
final class LocalFileServer {
    static let shared = LocalFileServer()

    private var listener: NWListener?
    private var port: UInt16?
    private var files: [String: URL] = [:]
    private var readyWaiters: [CheckedContinuation<UInt16?, Never>] = []

    /// Register `fileURL` and return a localhost URL that serves it (starting the
    /// server on first use). Returns nil if the server couldn't start.
    func servedURL(for fileURL: URL) async -> URL? {
        let name = fileURL.lastPathComponent
        files[name] = fileURL
        guard let port = await ensureStarted() else { return nil }
        let encoded = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        return URL(string: "http://localhost:\(port)/\(encoded)")
    }

    private func ensureStarted() async -> UInt16? {
        if let port { return port }
        if listener == nil { startListener() }
        return await withCheckedContinuation { continuation in
            if let port { continuation.resume(returning: port) }
            else { readyWaiters.append(continuation) }
        }
    }

    private func startListener() {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let listener = try? NWListener(using: params) else { resolveWaiters(nil); return }
        self.listener = listener
        listener.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                switch state {
                case .ready:
                    self.port = listener.port?.rawValue
                    self.resolveWaiters(self.port)
                case .failed, .cancelled:
                    self.listener = nil
                    self.resolveWaiters(nil)
                default:
                    break
                }
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            Task { @MainActor in self?.serve(connection) }
        }
        listener.start(queue: .main)
    }

    private func resolveWaiters(_ port: UInt16?) {
        let waiters = readyWaiters
        readyWaiters.removeAll()
        waiters.forEach { $0.resume(returning: port) }
    }

    // MARK: - Request handling

    private func serve(_ connection: NWConnection) {
        connection.start(queue: .main)
        // The request line + headers arrive in the first segment for a simple GET.
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, _, _ in
            Task { @MainActor in
                guard let self, let data, let request = String(data: data, encoding: .utf8) else {
                    connection.cancel(); return
                }
                self.respond(to: request, on: connection)
            }
        }
    }

    private func respond(to request: String, on connection: NWConnection) {
        // First line: "GET /name.ply HTTP/1.1"
        let path = request.split(separator: "\r\n").first?
            .split(separator: " ").dropFirst().first.map(String.init) ?? "/"
        let name = String(path.drop { $0 == "/" }).removingPercentEncoding ?? ""

        guard let fileURL = files[name], let body = try? Data(contentsOf: fileURL) else {
            send("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                 body: nil, on: connection)
            return
        }
        let header = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: application/octet-stream\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Access-Control-Allow-Origin: *\r\n"
            + "Cache-Control: no-store\r\n"
            + "Connection: close\r\n\r\n"
        send(header, body: body, on: connection)
    }

    private func send(_ header: String, body: Data?, on connection: NWConnection) {
        var response = Data(header.utf8)
        if let body { response.append(body) }
        connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
    }
}
