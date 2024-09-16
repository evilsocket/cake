FROM rust:bullseye as builder
RUN apt-get update && apt-get install -y libssl-dev ca-certificates cmake git
WORKDIR /app
ADD . /app
RUN cargo build --release

FROM debian:bullseye
RUN apt-get update && apt-get install -y libssl-dev ca-certificates
COPY --from=builder /app/target/release/cake-cli /usr/bin/cake-cli
COPY --from=builder /app/target/release/cake-split-model /usr/bin/cake-split-model
CMD ["/usr/bin/cake-cli"]