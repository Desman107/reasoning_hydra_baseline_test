/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hydra {

class CsvReader {
 public:
  // Construction.
  CsvReader() = default;
  explicit CsvReader(const std::string& file_name,
                     char separator = ',',
                     bool skip_first_line = true);
  virtual ~CsvReader() = default;

  // Setup.
  /**
   * @brief Setup the CsvReader by reading a csv file.
   * @param file_name The name of the csv file to read.
   * @param separator The separator used in the csv file.
   * @return True if the file was read successfully, false otherwise.
   */
  bool setup(const std::string& file_name,
             char separator = ',',
             bool skip_first_line = true);
  bool isSetup() const { return is_setup_; }
  operator bool() const { return isSetup(); }

  struct Row : std::vector<std::string> {
   public:
    // Construction.
    Row() = default;
    Row(const std::vector<std::string>& row,
        std::shared_ptr<std::unordered_map<std::string, size_t>> _header_to_index)
        : std::vector<std::string>(row), header_to_index(std::move(_header_to_index)) {}
    // Queries.
    const std::string& getEntry(const std::string& header) const;

   private:
    std::shared_ptr<std::unordered_map<std::string, size_t>> header_to_index;
  };

  // Queries.
  size_t numRows() const;
  bool hasHeader(const std::string& header) const;
  bool hasHeaders(const std::vector<std::string>& headers) const;
  const std::vector<std::string>& getHeaders() const;
  const std::vector<Row>& getRows() const;
  const Row& getRow(size_t row) const;
  const std::string& getEntry(const std::string& header, size_t row) const;

  // Interface to verify the read file with required and optional headers.
  /**
   * @brief Require the csv file to have the given headers. An error about missing
   * headers will be printed.
   * @param headers List of headers that are required.
   * @return True if all required headers are present, false otherwise.
   */
  bool checkRequiredHeaders(const std::vector<std::string>& headers) const;

  /**
   * @brief Check if the csv file has the given headers. A warning about missing headers
   * will be printed.
   * @param headers List of headers that are optional.
   */
  void checkOptionalHeaders(const std::vector<std::string>& headers) const;

 private:
  // State.
  bool is_setup_ = false;
  std::string loaded_file_name_;

  // Data.
  std::vector<std::string> headers_;
  std::vector<Row> rows_;
  std::shared_ptr<std::unordered_map<std::string, size_t>> header_to_index_;

  // Helper functions.
  std::vector<std::string> missingHeaders(
      const std::vector<std::string>& headers) const;
};

}  // namespace hydra
