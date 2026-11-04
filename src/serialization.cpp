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
        if (position_ == source_.size() ||
            source_[position_] < '0' || source_[position_] > '9')
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

Json const& field(Json const& value, std::string const& key) {
    if (value.kind != Json::Kind::Object)
        throw std::invalid_argument("Expected JSON object");
    const auto it = value.object.find(key);
    if (it == value.object.end())
        throw std::invalid_argument("Missing required JSON field: " + key);
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
    if (number > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument("JSON dimension not representable");
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
    if (end < 0 || static_cast<std::uintmax_t>(end) >
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
std::uint64_t decode64(std::string_view s) {
    if (s.size()!=8u) throw std::invalid_argument("Expected eight bytes");
    std::uint64_t result=0u;
    for (unsigned i=0u;i<8u;++i)
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

} // namespace

void neural_save_snapshot(NeuralParameterSnapshot const& snapshot,
                          std::string const& manifest_path,
                          std::string const& safetensors_path) {
    if (manifest_path.empty() || safetensors_path.empty() ||
        manifest_path == safetensors_path)
        throw std::invalid_argument("Snapshot output paths must be distinct");
    const auto& network = snapshot.network();
    ensure_limit(network);
    std::string manifest = "{\"format\":\"ariadne-neural\",\"version\":1,"
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
    manifest += "],\"tensors\":\"safetensors\"}\n";
    header += "}";
    if (manifest.size()>max_manifest_bytes ||
        header.size()>max_header_bytes || payload.size()>max_payload_bytes)
        throw std::invalid_argument("Neural snapshot exceeds file size limit");
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
    if (integer(field(manifest,"version")) != 1u)
        throw std::invalid_argument("Unsupported neural manifest version");
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
    if (next!=binary.size()-data_begin ||
        next!=network.parameter_count()*8u)
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
    return NeuralParameterSnapshot(std::move(network),std::move(values));
}

} // namespace Ariadne
