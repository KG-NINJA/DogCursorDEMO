/*! coi-serviceworker v0.1.7 - Guido Zuidhof, licensed under MIT */
let coepCredentialless = false;
if (typeof window === 'undefined') {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

    self.addEventListener("message", (ev) => {
        if (!ev.data) {
            return;
        } else if (ev.data.type === "deregister") {
            self.registration.unregister().then(() => {
                return self.clients.matchAll();
            }).then(clients => {
                clients.forEach((client) => client.navigate(client.url));
            });
        } else if (ev.data.type === "coepCredentialless") {
            coepCredentialless = ev.data.value;
        }
    });

    self.addEventListener("fetch", function (event) {
        const r = event.request;
        if (r.cache === "only-if-cached" && r.mode !== "same-origin") {
            return;
        }

        const coep = coepCredentialless ? "credentialless" : "require-corp";

        event.respondWith(
            fetch(r).then((response) => {
                if (response.status === 0) {
                    return response;
                }

                const newHeaders = new Headers(response.headers);
                newHeaders.set("Cross-Origin-Embedder-Policy", coep);
                newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

                return new Response(response.body, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: newHeaders,
                });
            })
        );
    });

} else {
    (async function () {
        if (window.crossOriginIsolated !== false) return;

        const libraryPath = document.currentScript.src;
        const path = libraryPath.substring(0, libraryPath.lastIndexOf('/') + 1);
        const workerPath = path + "coi-serviceworker.js"; // Standard path

        try {
            const registration = await navigator.serviceWorker.register(workerPath);
            console.log("COI Service Worker registered", registration);

            registration.addEventListener("updatefound", () => {
                console.log("COI: Reloading to activate service worker");
                window.location.reload();
            });

            if (registration.active && !navigator.serviceWorker.controller) {
                console.log("COI: Reloading to control page");
                window.location.reload();
            }

        } catch (err) {
            console.error("COI Service Worker failed to register:", err);
        }
    })();
}
