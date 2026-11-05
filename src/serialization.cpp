/***************************************************************************
 *            src/serialization.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neural.
 *
 *  Ariadne Neural is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neural is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neural.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "neural/serialization.hpp"
#include "neural/sha256.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ariadne {
namespace {

constexpr std::size_t max_manifest_bytes = 1024u * 1024u;
constexpr std::size_t max_header_bytes = 1024u * 1024u;
constexpr std::size_t max_payload_bytes = 256u * 1024u * 1024u;
constexpr std::size_t max_layers = 4096u;
static_assert(sizeof(double) == 8u && sizeof(std::uint64_t) == 8u);
static_assert(std::numeric_limits<double>::is_iec559,
              "Safetensors F64 requires IEEE-754 binary64");

struct Json {
    enum class Kind { Number, String, Array, Object };
    Kind kind = Kind::Number;
    std::uint64_t number = 0u;
    std::string string;
    std::vector<Json> array;
    std::map<std::string,Json> object;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view source) : source_(source) {}
    Json parse() {
        auto result = value(0u);
        spaces();
        if (position_ != source_.size())
            fail("Trailing content after JSON");
        return result;
    }
private:
    std::string_view source_;
    std::size_t position_ = 0u;

    [[noreturn]] void fail(char const* message) const {
        throw std::invalid_argument(message);
    }

    void spaces() {
        while (position_ < source_.size() &&
               (source_[position_] == ' ' || source_[position_] == '\n' ||
                source_[position_] == '\r' || source_[position_] == '\t'))
            ++position_;
    }
    bool consume(char c) {
        spaces();
        if (position_ < source_.size() && source_[position_] == c) {
            ++position_;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!consume(c)) fail("Unexpected JSON token");
    }
    std::string string() {
        expect('"');
        std::string output;
        while (position_ < source_.size()) {
            const unsigned char c =
                static_cast<unsigned char>(source_[position_++]);
            if (c == '"') return output;
            if (c == '\\' || c < 0x20u || c >= 0x7fu)
                fail("JSON strings must use unescaped printable ASCII");
            output.push_back(static_cast<char>(c));
        }
        fail("Unterminated JSON string");
    }
    std::uint64_t number() {
        spaces();
        // value() guarantees position_ < source_.size() before calling this.
        if (source_[position_] < '0' || source_[position_] > '9')
            fail("Expected unsigned integer");
        if (source_[position_] == '0' && position_ + 1u < source_.size() &&
            source_[position_+1u] >= '0' && source_[position_+1u] <= '9')
            fail("Noncanonical JSON integer");
        std::uint64_t result = 0u;
        while (position_ < source_.size() &&
               source_[position_] >= '0' && source_[position_] <= '9') {
            const auto digit = static_cast<unsigned>(source_[position_] - '0');
            if (result > (std::numeric_limits<std::uint64_t>::max()-digit)/10u)
                fail("JSON integer overflow");
            result = result * 10u + digit;
            ++position_;
        }
        return result;
    }
    Json value(unsigned depth) {
        if (depth > 16u) fail("JSON nesting too deep");
        spaces();
        if (position_ >= source_.size()) fail("Unexpected JSON end");
        Json result;
        if (source_[position_] == '"') {
            result.kind = Json::Kind::String;
            result.string = string();
        } else if (source_[position_] == '[') {
            result.kind = Json::Kind::Array;
            expect('[');
            if (!consume(']')) {
                do {
                    result.array.push_back(value(depth+1u));
                    if (result.array.size() > max_layers*4u)
                        fail("JSON array too large");
                    if (consume(']')) break;
                    expect(',');
                } while (true);
            }
        } else if (source_[position_] == '{') {
            result.kind = Json::Kind::Object;
            expect('{');
            if (!consume('}')) {
                do {
                    const std::string key = string();
                    expect(':');
                    auto item = value(depth+1u);
                    if (!result.object.emplace(key,std::move(item)).second)
                        fail("Duplicate JSON key");
                    if (result.object.size() > max_layers*4u)
                        fail("JSON object too large");
                    if (consume('}')) break;
                    expect(',');
                } while (true);
            }
        } else {
            result.kind = Json::Kind::Number;
            result.number = number();
        }
        return result;
    }
};

// Every caller invokes keys() on the same value first, which already
// guarantees Kind::Object; a non-object therefore simply misses the key.
Json const& field(Json const& value, char const* key) {
    const auto it = value.object.find(key);
    if (it == value.object.end())
        throw std::invalid_argument(std::string("Missing required JSON field: ") + key);
    return it->second;
}
void keys(Json const& j, std::size_t expected) {
    if (j.kind != Json::Kind::Object || j.object.size() != expected)
        throw std::invalid_argument("Unknown or missing JSON fields");
}
std::uint64_t integer(Json const& j) {
    if (j.kind != Json::Kind::Number)
        throw std::invalid_argument("Expected JSON integer");
    return j.number;
}
std::string const& text(Json const& j) {
    if (j.kind != Json::Kind::String)
        throw std::invalid_argument("Expected JSON string");
    return j.string;
}
std::size_t size(Json const& j) {
    const auto number = integer(j);
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (number > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()))
            throw std::invalid_argument("JSON dimension not representable");
    }
    return static_cast<std::size_t>(number);
}
std::string const& literal(Json const& j, char const* expected) {
    const std::string& actual = text(j);
    if (actual != expected)
        throw std::invalid_argument("Unsupported serialization format or type");
    return actual;
}
std::string read_file(std::string const& path, std::size_t limit) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open neural snapshot file");
    const std::streamoff end = stream.tellg();
    // A negative tellg() converts to a huge unsigned value and is rejected
    // by the same limit check.
    if (static_cast<std::uintmax_t>(end) >
                  static_cast<std::uintmax_t>(limit))
        throw std::invalid_argument("Neural snapshot file exceeds size limit");
    std::string result(static_cast<std::size_t>(end),'\0');
    stream.seekg(0);
    if (!result.empty() && !stream.read(result.data(),
                static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("Cannot read neural snapshot file");
    return result;
}
void write_file(std::string const& path, std::string const& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot open neural snapshot output");
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write neural snapshot output");
}
void append64(std::string& result, std::uint64_t n) {
    for (unsigned i=0u;i<8u;++i) {
        result.push_back(static_cast<char>(n & 0xffu));
        n >>= 8u;
    }
}
// Every caller passes a substring of at most eight bytes taken from a buffer
// whose length has already been validated, so no length check is needed.
std::uint64_t decode64(std::string_view s) {
    std::uint64_t result=0u;
    for (unsigned i=0u;i<s.size();++i)
        result |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(s[i])) << (i*8u);
    return result;
}
std::string activation(NeuralActivation a) {
    return std::string(neural_activation_name(a));
}
NeuralActivation parse_activation(std::string const& a) {
    for (auto v : {NeuralActivation::Identity,NeuralActivation::Relu,
                   NeuralActivation::Tanh,NeuralActivation::Sigmoid,
                   NeuralActivation::Softplus})
        if (activation(v)==a) return v;
    throw std::invalid_argument("Unsupported neural activation");
}
void ensure_limit(NeuralNetwork const& network) {
    if (network.layers().size() > max_layers ||
        network.parameter_count() > max_payload_bytes/8u)
        throw std::invalid_argument("Neural snapshot exceeds resource limit");
}
std::string name(std::size_t layer, char const* suffix) {
    return "layers." + std::to_string(layer) + "." + suffix;
}

struct Descriptor {
    std::vector<std::size_t> shape;
    std::size_t begin;
    std::size_t end;
};

void add_header_tensor(std::string& header, bool& first,
                       std::string const& tensor_name,
                       std::vector<std::size_t> const& shape,
                       std::size_t& cursor) {
    if (!first) header.push_back(',');
    first = false;
    const std::size_t entries = shape.size() == 2u
        ? shape[0] * shape[1] : shape[0];
    const std::size_t end = cursor + entries*8u;
    header += "\"" + tensor_name + "\":{\"dtype\":\"F64\",\"shape\":[";
    for (std::size_t i=0u;i<shape.size();++i) {
        if (i!=0u) header.push_back(',');
        header += std::to_string(shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(cursor) + ","
              + std::to_string(end) + "]}";
    cursor = end;
}
Descriptor descriptor(Json const& j) {
    keys(j,3u);
    (void)literal(field(j,"dtype"),"F64");
    const Json& shape = field(j,"shape");
    const Json& offsets = field(j,"data_offsets");
    if (shape.kind!=Json::Kind::Array || offsets.kind!=Json::Kind::Array ||
        offsets.array.size()!=2u ||
        (shape.array.size()!=1u && shape.array.size()!=2u))
        throw std::invalid_argument("Invalid Safetensors shape or offsets");
    Descriptor result{{},size(offsets.array[0]),size(offsets.array[1])};
    for (auto const& dim : shape.array) {
        const std::size_t n = size(dim);
        if (n==0u) throw std::invalid_argument("Empty Safetensors dimension");
        result.shape.push_back(n);
    }
    std::size_t elements=1u;
    for (std::size_t n : result.shape) {
        if (elements > max_payload_bytes / 8u / n)
            throw std::invalid_argument("Safetensors shape exceeds limit");
        elements *= n;
    }
    if (result.end < result.begin ||
        result.end-result.begin != elements*8u)
        throw std::invalid_argument("Safetensors payload size mismatch");
    return result;
}
void verify_tensor(std::map<std::string,Json> const& tensors,
                   std::string const& tensor_name,
                   std::vector<std::size_t> const& expected_shape,
                   std::size_t& offset) {
    const auto it = tensors.find(tensor_name);
    if (it == tensors.end()) throw std::invalid_argument("Missing Safetensors tensor");
    const auto d = descriptor(it->second);
    if (d.shape != expected_shape || d.begin != offset)
        throw std::invalid_argument("Incorrect Safetensors tensor shape or order");
    offset = d.end;
}

std::string hex_bits(double value) {
    constexpr char digits[] = "0123456789abcdef";
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::string result(16u,'0');
    for (unsigned i = 0; i < 16u; ++i)
        result[i] = digits[(bits >> (4u*(15u-i))) & 0xfu];
    return result;
}

std::uint64_t decode_hex(std::string const& value) {
    if (value.size() != 16u)
        throw std::invalid_argument("Wrong neural optimizer numeric encoding");
    std::uint64_t n=0u;
    for (char c : value) {
        unsigned digit;
        if (c >= '0' && c <= '9')
            digit = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f')
            digit = static_cast<unsigned>(c - 'a') + 10u;
        else
            throw std::invalid_argument("Invalid neural optimizer numeric encoding");
        n = (n << 4u) | static_cast<std::uint64_t>(digit);
    }
    return n;
}

double decode_double(std::string const& value) {
    return std::bit_cast<double>(decode_hex(value));
}

std::size_t decimal_size(std::string const& s) {
    if (s.empty() || (s.size()>1u && s[0]=='0'))
        throw std::invalid_argument("Invalid neural optimizer step count");
    std::size_t result=0u;
    for (char c : s) {
        if (c<'0' || c>'9' ||
            result > (std::numeric_limits<std::size_t>::max()
                     -static_cast<unsigned>(c-'0'))/10u)
            throw std::invalid_argument("Invalid neural optimizer step count");
        result = result*10u + static_cast<unsigned>(c-'0');
    }
    return result;
}

std::string mask_text(std::vector<bool> const& mask) {
    std::string result;
    result.reserve(mask.size());
    for (bool enabled : mask)
        result.push_back(enabled ? '1' : '0');
    return result;
}

std::vector<bool> parse_mask(std::string const& s, std::size_t layers) {
    if (s.size()!=layers)
        throw std::invalid_argument("Wrong neural training mask dimension");
    std::vector<bool> result;
    result.reserve(s.size());
    for (char c : s) {
        if (c!='0' && c!='1')
            throw std::invalid_argument("Invalid neural training mask");
        result.push_back(c=='1');
    }
    return result;
}

// The session constructor validates the optimizer kind, so the fallback is
// exactly AdamW.
char const* optimizer_name(NeuralOptimizerKind kind) {
    return kind == NeuralOptimizerKind::Sgd ? "sgd" : "adamw";
}

NeuralOptimizerKind parse_optimizer(std::string const& name) {
    if (name=="sgd") return NeuralOptimizerKind::Sgd;
    if (name=="adamw") return NeuralOptimizerKind::AdamW;
    throw std::invalid_argument("Unsupported neural optimizer kind");
}

// The session constructor validates the loss kind, so the fallback is
// exactly BinaryCrossEntropyWithLogits.
char const* loss_name(NeuralLossKind kind) {
    return kind == NeuralLossKind::MeanSquaredError ? "mse" : "bce_logits";
}

NeuralLossKind parse_loss(std::string const& name) {
    if (name=="mse") return NeuralLossKind::MeanSquaredError;
    if (name=="bce_logits") return NeuralLossKind::BinaryCrossEntropyWithLogits;
    throw std::invalid_argument("Unsupported neural loss kind");
}

std::string training_sha256(NeuralTrainingCheckpoint const& checkpoint) {
    Detail::Sha256 sha;
    constexpr char domain[] = "ariadne-neural-training-state";
    sha.update(std::string_view(domain,sizeof(domain)));
    sha.update(neural_snapshot_sha256(checkpoint.snapshot));
    sha.update_u64(checkpoint.options.kind==NeuralOptimizerKind::Sgd ? 0u : 1u);
    sha.update_u64(checkpoint.loss==NeuralLossKind::MeanSquaredError ? 0u : 1u);
    auto const& o = checkpoint.options;
    for (double value : {o.learning_rate,o.momentum,o.weight_decay,
                         o.beta1,o.beta2,o.epsilon})
        sha.update_u64(std::bit_cast<std::uint64_t>(value));
    sha.update_u64(static_cast<std::uint64_t>(checkpoint.steps));
    sha.update_u64(static_cast<std::uint64_t>(checkpoint.trainable_layers.size()));
    const auto mask=mask_text(checkpoint.trainable_layers);
    sha.update(mask);
    // Iterate over the moment buffers without copying either large vector.
    for (auto const* moment : {&checkpoint.first_moment, &checkpoint.second_moment}) {
        sha.update_u64(static_cast<std::uint64_t>(moment->size()));
        for (double value : *moment)
            sha.update_u64(std::bit_cast<std::uint64_t>(value));
    }
    return sha.hex_digest();
}


} // namespace

std::string neural_snapshot_sha256(NeuralParameterSnapshot const& snapshot) {
    // Domain-separated canonical binary representation, independent of
    // JSON formatting, Safetensors headers and host byte order.
    Detail::Sha256 sha;
    constexpr char domain[] = "ariadne-neural-model";
    sha.update(std::string_view(domain, sizeof(domain))); // Include NUL.
    const auto& network = snapshot.network();
    sha.update_u64(static_cast<std::uint64_t>(network.input_dimension()));
    sha.update_u64(static_cast<std::uint64_t>(network.layers().size()));
    // NeuralNetwork validates every layer activation at construction.
    static constexpr unsigned char codes[] = {0u, 1u, 2u, 3u, 4u};
    for (auto const& layer : network.layers()) {
        sha.update_u64(static_cast<std::uint64_t>(layer.outputs));
        const char encoded = static_cast<char>(
            codes[static_cast<unsigned>(layer.activation)]);
        sha.update(std::string_view(&encoded, 1u));
    }
    for (double value : snapshot.values())
        sha.update_u64(std::bit_cast<std::uint64_t>(value));
    return sha.hex_digest();
}

void neural_save_snapshot(NeuralParameterSnapshot const& snapshot,
                          std::string const& manifest_path,
                          std::string const& safetensors_path) {
    if (manifest_path.empty() || safetensors_path.empty() ||
        manifest_path == safetensors_path)
        throw std::invalid_argument("Snapshot output paths must be distinct");
    const auto& network = snapshot.network();
    ensure_limit(network);
    std::string manifest = "{\"format\":\"ariadne-neural\","
        "\"numeric\":\"IEEE754-binary64\",\"input_dimension\":"
        + std::to_string(network.input_dimension()) + ",\"layers\":[";
    std::string header = "{";
    std::string payload;
    payload.reserve(network.parameter_count()*8u);
    std::size_t cursor=0u;
    bool first=true;
    auto const& values=snapshot.values();
    for (std::size_t i=0u;i<network.layers().size();++i) {
        auto const& layer=network.layers()[i];
        if (i!=0u) manifest.push_back(',');
        manifest += "{\"outputs\":" + std::to_string(layer.outputs)
            + ",\"activation\":\"" + activation(layer.activation) + "\"}";
        add_header_tensor(header,first,name(i,"weight"),
                          {layer.outputs,layer.inputs},cursor);
        add_header_tensor(header,first,name(i,"bias"),{layer.outputs},cursor);
        for (std::size_t n=0u;n<layer.outputs*layer.inputs;++n)
            append64(payload,std::bit_cast<std::uint64_t>(
                values[layer.weights_offset+n]));
        for (std::size_t n=0u;n<layer.outputs;++n)
            append64(payload,std::bit_cast<std::uint64_t>(
                values[layer.biases_offset+n]));
    }
    manifest += "],\"tensors\":\"safetensors\",\"snapshot_sha256\":\"" +
        neural_snapshot_sha256(snapshot) + "\"}\n";
    header += "}";
    // ensure_limit already bounds manifest (<1MB), header (<1MB) and payload
    // (<=max_payload_bytes), so no separate file size check is needed here.
    std::string binary;
    binary.reserve(8u+header.size()+payload.size());
    append64(binary,static_cast<std::uint64_t>(header.size()));
    binary+=header;
    binary+=payload;
    // Two distinct files; callers publish them together after writing.
    write_file(safetensors_path,binary);
    write_file(manifest_path,manifest);
}

NeuralParameterSnapshot neural_load_snapshot(
    std::string const& manifest_path,
    std::string const& safetensors_path) {
    if (manifest_path.empty() || safetensors_path.empty() ||
        manifest_path==safetensors_path)
        throw std::invalid_argument("Snapshot input paths must be distinct");
    const auto manifest_bytes = read_file(manifest_path,max_manifest_bytes);
    auto manifest = JsonParser(manifest_bytes).parse();
    keys(manifest,6u);
    (void)literal(field(manifest,"format"),"ariadne-neural");
    (void)literal(field(manifest,"numeric"),"IEEE754-binary64");
    (void)literal(field(manifest,"tensors"),"safetensors");
    const std::size_t inputs = size(field(manifest,"input_dimension"));
    const Json& layers = field(manifest,"layers");
    if (layers.kind != Json::Kind::Array ||
        layers.array.empty() || layers.array.size()>max_layers)
        throw std::invalid_argument("Invalid neural manifest layers");
    std::vector<NeuralDenseLayerSpec> specifications;
    specifications.reserve(layers.array.size());
    for (auto const& layer : layers.array) {
        keys(layer,2u);
        specifications.push_back({size(field(layer,"outputs")),
                                  parse_activation(text(field(layer,"activation")))});
    }
    NeuralNetwork network(inputs,std::move(specifications));
    ensure_limit(network);
    const auto binary=read_file(safetensors_path,
        8u+max_header_bytes+max_payload_bytes);
    if (binary.size()<8u)
        throw std::invalid_argument("Truncated Safetensors header length");
    const auto header_size=decode64(std::string_view(binary).substr(0u,8u));
    if (header_size==0u || header_size>max_header_bytes ||
        header_size>binary.size()-8u)
        throw std::invalid_argument("Invalid Safetensors header length");
    const std::size_t data_begin=8u+static_cast<std::size_t>(header_size);
    const Json metadata=JsonParser(
        std::string_view(binary).substr(8u,static_cast<std::size_t>(header_size))).parse();
    if (metadata.kind != Json::Kind::Object ||
        metadata.object.size()!=network.layers().size()*2u)
        throw std::invalid_argument("Unexpected Safetensors tensor collection");
    std::size_t next=0u;
    for (std::size_t i=0u;i<network.layers().size();++i) {
        auto const& layer=network.layers()[i];
        verify_tensor(metadata.object,name(i,"weight"),
                      {layer.outputs,layer.inputs},next);
        verify_tensor(metadata.object,name(i,"bias"),{layer.outputs},next);
    }
    // verify_tensor already matched every tensor shape and offset against
    // the network layout, so next == parameter_count()*8u is guaranteed here.
    if (next!=binary.size()-data_begin)
        throw std::invalid_argument("Safetensors trailing or missing payload bytes");
    std::vector<double> values(network.parameter_count());
    std::size_t cursor=0u;
    for (std::size_t i=0u;i<network.layers().size();++i) {
        auto const& layer=network.layers()[i];
        for (std::size_t n=0u;n<layer.outputs*layer.inputs;++n) {
            values[layer.weights_offset+n]=std::bit_cast<double>(
                decode64(std::string_view(binary).substr(data_begin+cursor,8u)));
            cursor+=8u;
        }
        for (std::size_t n=0u;n<layer.outputs;++n) {
            values[layer.biases_offset+n]=std::bit_cast<double>(
                decode64(std::string_view(binary).substr(data_begin+cursor,8u)));
            cursor+=8u;
        }
    }
    NeuralParameterSnapshot snapshot(std::move(network),std::move(values));
    const auto& saved_digest = text(field(manifest,"snapshot_sha256"));
    if (saved_digest.size() != 64u ||
        saved_digest.find_first_not_of("0123456789abcdef") != std::string::npos ||
        saved_digest != neural_snapshot_sha256(snapshot))
        throw std::invalid_argument("Neural snapshot SHA-256 identity mismatch");
    return snapshot;
}

void neural_save_training_checkpoint(
    NeuralTrainingSession const& session,
    std::string const& manifest_path,
    std::string const& safetensors_path,
    std::string const& optimizer_path) {
    if (optimizer_path.empty() || optimizer_path == manifest_path ||
        optimizer_path == safetensors_path ||
        manifest_path.empty() || safetensors_path.empty() ||
        manifest_path == safetensors_path)
        throw std::invalid_argument("Training checkpoint paths must be distinct");
    auto checkpoint = session.export_checkpoint();
    ensure_limit(checkpoint.snapshot.network());
    const auto count=checkpoint.snapshot.network().parameter_count();
    const auto byte_count=count*8u;
    const auto& o=checkpoint.options;
    std::string header =
        R"({"__metadata__":{"format":"ariadne-neural-training","snapshot_sha256":")"
        + neural_snapshot_sha256(checkpoint.snapshot)
        + R"(","checkpoint_sha256":")" + training_sha256(checkpoint)
        + R"(","optimizer":")" + optimizer_name(o.kind)
        + R"(","loss":")" + loss_name(checkpoint.loss)
        + R"(","steps":")" + std::to_string(checkpoint.steps)
        + R"(","trainable_layers":")" + mask_text(checkpoint.trainable_layers)
        + R"(","learning_rate":")" + hex_bits(o.learning_rate)
        + R"(","momentum":")" + hex_bits(o.momentum)
        + R"(","weight_decay":")" + hex_bits(o.weight_decay)
        + R"(","beta1":")" + hex_bits(o.beta1)
        + R"(","beta2":")" + hex_bits(o.beta2)
        + R"(","epsilon":")" + hex_bits(o.epsilon)
        + R"("},"first_moment":{"dtype":"F64","shape":[)"
        + std::to_string(count)
        + R"(],"data_offsets":[0,)" + std::to_string(byte_count)
        + R"(]},"second_moment":{"dtype":"F64","shape":[)"
        + std::to_string(count)
        + R"(],"data_offsets":[)" + std::to_string(byte_count)
        + "," + std::to_string(byte_count*2u) + "]}}";
    // The header is a fixed-size template: count is bounded by ensure_limit,
    // so it can never exceed max_header_bytes.
    std::string binary;
    binary.reserve(8u+header.size()+byte_count*2u);
    append64(binary,static_cast<std::uint64_t>(header.size()));
    binary += header;
    for (double value : checkpoint.first_moment)
        append64(binary,std::bit_cast<std::uint64_t>(value));
    for (double value : checkpoint.second_moment)
        append64(binary,std::bit_cast<std::uint64_t>(value));
    // Three files are individually written but not atomically published.
    // The model+state identity checks detect cross-file mismatches on load.
    neural_save_snapshot(checkpoint.snapshot,manifest_path,safetensors_path);
    write_file(optimizer_path,binary);
}

NeuralTrainingSession neural_load_training_checkpoint(
    std::string const& manifest_path,
    std::string const& safetensors_path,
    std::string const& optimizer_path) {
    if (optimizer_path.empty() || optimizer_path == manifest_path ||
        optimizer_path == safetensors_path ||
        manifest_path.empty() || safetensors_path.empty() ||
        manifest_path == safetensors_path)
        throw std::invalid_argument("Training checkpoint paths must be distinct");
    auto snapshot=neural_load_snapshot(manifest_path,safetensors_path);
    auto const count=snapshot.network().parameter_count();
    auto const binary=read_file(optimizer_path,
        8u+max_header_bytes+2u*max_payload_bytes);
    if (binary.size()<8u)
        throw std::invalid_argument("Truncated training Safetensors header");
    const auto header_size=decode64(std::string_view(binary).substr(0u,8u));
    if (header_size==0u || header_size>max_header_bytes ||
        header_size>binary.size()-8u)
        throw std::invalid_argument("Invalid training Safetensors header length");
    const std::size_t data_begin=8u+static_cast<std::size_t>(header_size);
    const auto json=JsonParser(
        std::string_view(binary).substr(8u,static_cast<std::size_t>(header_size))).parse();
    keys(json,3u);
    auto const& metadata=field(json,"__metadata__");
    keys(metadata,13u);
    (void)literal(field(metadata,"format"),"ariadne-neural-training");
    if (text(field(metadata,"snapshot_sha256"))!=neural_snapshot_sha256(snapshot))
        throw std::invalid_argument("Training state refers to a different model");
    const auto saved_hash=text(field(metadata,"checkpoint_sha256"));
    const auto steps=decimal_size(text(field(metadata,"steps")));
    const auto mask=parse_mask(text(field(metadata,"trainable_layers")),
                              snapshot.network().layers().size());
    NeuralOptimizerOptions options;
    options.kind=parse_optimizer(text(field(metadata,"optimizer")));
    options.learning_rate=decode_double(text(field(metadata,"learning_rate")));
    options.momentum=decode_double(text(field(metadata,"momentum")));
    options.weight_decay=decode_double(text(field(metadata,"weight_decay")));
    options.beta1=decode_double(text(field(metadata,"beta1")));
    options.beta2=decode_double(text(field(metadata,"beta2")));
    options.epsilon=decode_double(text(field(metadata,"epsilon")));
    auto loss=parse_loss(text(field(metadata,"loss")));
    std::size_t offset=0u;
    verify_tensor(json.object,"first_moment",{count},offset);
    verify_tensor(json.object,"second_moment",{count},offset);
    if (offset!=count*16u || offset!=binary.size()-data_begin)
        throw std::invalid_argument("Wrong training state Safetensors length");
    std::vector<double> first(count),second(count);
    for (std::size_t i=0u;i<count;++i) {
        first[i]=std::bit_cast<double>(decode64(
            std::string_view(binary).substr(data_begin+8u*i,8u)));
        second[i]=std::bit_cast<double>(decode64(
            std::string_view(binary).substr(data_begin+8u*count+8u*i,8u)));
    }
    NeuralTrainingSession result(NeuralTrainingCheckpoint{
        std::move(snapshot), options, loss, mask,
        std::move(first), std::move(second), steps
    });
    const auto actual_hash=training_sha256(result.export_checkpoint());
    if (saved_hash.size()!=64u ||
        saved_hash.find_first_not_of("0123456789abcdef")!=std::string::npos ||
        saved_hash!=actual_hash)
        throw std::invalid_argument("Training checkpoint SHA-256 mismatch");
    return result;
}

} // namespace Ariadne
