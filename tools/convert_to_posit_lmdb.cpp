/*
 * convert_to_posit_lmdb.cpp
 *
 *  Created on: Nov 2, 2019
 *      Author: himeshi
 */

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/fp16.hpp"
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

#define CIFAR10

#define TEST 0
uint64_t offset_;
scoped_ptr<db::Cursor> cursor_read;

#ifdef CIFAR10
BlobProto data_mean;
#endif

bool Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank || TEST;
  return !keep;
}

string int_array_to_string(fp16 int_array[], int size_of_array) {
	ostringstream oss("");
	for (int temp = 0; temp < size_of_array; temp++)
		oss << int_array[temp];
	return oss.str();
}

void convert_train_db() {
	scoped_ptr<db::DB> db_read(db::GetDB (FLAGS_backend));
#ifdef MNIST
	db_read->Open("/home/himeshi/dl/caffe-stripped/examples/mnist/mnist_train_lmdb", db::READ);
#else
#ifdef CIFAR10
	db_read->Open("/home/himeshi/dl/caffe-stripped/examples/cifar10/cifar10_train_lmdb", db::READ);
#else
	printf("ERROR: Unkown database.\n");
#endif
#endif
	scoped_ptr<db::Cursor> cursor_read(db_read->NewCursor());

	scoped_ptr<db::DB> db_write(db::GetDB (FLAGS_backend));
	std::ostringstream s;
#ifdef MNIST
	s << "/home/himeshi/dl/caffe-stripped/examples/mnist/mnist_train_posit_" << _G_NBITS << "_" << _G_ESIZE << "_lmdb";
#else
#ifdef CIFAR10
	s << "/home/himeshi/dl/caffe-stripped/examples/cifar10/cifar10_train_posit_" << _G_NBITS << "_" << _G_ESIZE << "_lmdb";
#else
	printf("ERROR: Unkown database.\n");
#endif
#endif
	db_write->Open(s.str(), db::NEW);
	scoped_ptr<db::Transaction> txn(db_write->NewTransaction());
	scoped_ptr<db::Cursor> cursor_write(db_write->NewCursor());

	Datum datum;
	DatumNew datumNew;
	int count = 0;
	while(cursor_read->valid()) {
		datum.ParseFromString(cursor_read->value());

		const int datum_channels = datum.channels();
		const int datum_height = datum.height();
		const int datum_width = datum.width();

		datumNew.set_channels(datum_channels);
		datumNew.set_height(datum_height);
		datumNew.set_width(datum_width);
		datumNew.set_label(fp32tofp16(datum.label()));
		//datumNew.set_float_data(datum.float_data());
		datumNew.set_encoded(datum.encoded());

		int size = datum_channels * datum_width * datum_height;
		uint16_t* dataNew = (uint16_t*) malloc(size * sizeof(uint16_t));
		const string& data = datum.data();

		int data_index;
		float datum_element;
		for (int c = 0; c < datum_channels; ++c) {
			for (int h = 0; h < datum_height; ++h) {
				for (int w = 0; w < datum_width; ++w) {
					data_index = (c * datum_height + h) * datum_width + w;
#ifdef MNIST
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index])) * 0.00390625;
#else
#ifdef CIFAR10
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index])) - data_mean.data(data_index);
#else
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
#endif
#endif

					dataNew[data_index] = (uint16_t) fp32tofp16(datum_element);
				}
			}
		}

		char* dataNewBytes = (char*) malloc(size * sizeof(uint16_t));
		for (int j = 0; j < size * 2; j += 2) {
			dataNewBytes[j] = dataNew[j / 2] >> 8;
			dataNewBytes[j + 1] = dataNew[j / 2] & 0x00FF;
		}
		datumNew.set_data(dataNewBytes, size * 2);

		string output;
		datumNew.SerializeToString(&output);

		txn->Put(cursor_read->key(), output);

		free(dataNew);
		free(dataNewBytes);
		count++;
		cursor_read->Next();
	}
	txn->Commit();
	printf("\n%d\n\n", count);
}

void convert_test_db() {
	scoped_ptr<db::DB> db_read(db::GetDB (FLAGS_backend));
#ifdef MNIST
	db_read->Open("/home/himeshi/dl/caffe-stripped/examples/mnist/mnist_test_lmdb", db::READ);
#else
#ifdef CIFAR10
	db_read->Open("/home/himeshi/dl/caffe-stripped/examples/cifar10/cifar10_test_lmdb", db::READ);
#else
	printf("ERROR: Unkown database.\n");
#endif
#endif
	scoped_ptr<db::Cursor> cursor_read(db_read->NewCursor());

	scoped_ptr<db::DB> db_write(db::GetDB (FLAGS_backend));
	std::ostringstream s;
#ifdef MNIST
	s << "/home/himeshi/dl/caffe-stripped/examples/mnist/mnist_test_posit_" << _G_NBITS << "_" << _G_ESIZE << "_lmdb";
#else
#ifdef CIFAR10
	s << "/home/himeshi/dl/caffe-stripped/examples/cifar10/cifar10_test_posit_" << _G_NBITS << "_" << _G_ESIZE << "_lmdb";
#else
	printf("ERROR: Unkown database.\n");
#endif
#endif
	db_write->Open(s.str(), db::NEW);
	scoped_ptr<db::Transaction> txn(db_write->NewTransaction());
	scoped_ptr<db::Cursor> cursor_write(db_write->NewCursor());

	Datum datum;
	DatumNew datumNew;
	int count = 0;
	while(cursor_read->valid()) {
		datum.ParseFromString(cursor_read->value());

		const int datum_channels = datum.channels();
		const int datum_height = datum.height();
		const int datum_width = datum.width();

		datumNew.set_channels(datum_channels);
		datumNew.set_height(datum_height);
		datumNew.set_width(datum_width);
		datumNew.set_label(fp32tofp16(datum.label()));
		//datumNew.set_float_data(datum.float_data());
		datumNew.set_encoded(datum.encoded());

		int size = datum_channels * datum_width * datum_height;
		uint16_t* dataNew = (uint16_t*) malloc(size * sizeof(uint16_t));
		const string& data = datum.data();

		int data_index;
		float datum_element;
		for (int c = 0; c < datum_channels; ++c) {
			for (int h = 0; h < datum_height; ++h) {
				for (int w = 0; w < datum_width; ++w) {
					data_index = (c * datum_height + h) * datum_width + w;
#ifdef MNIST
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index])) * 0.00390625;
#else
#ifdef CIFAR10
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index])) - data_mean.data(data_index);
#else
					datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
#endif
#endif
					dataNew[data_index] = (uint16_t) fp32tofp16(datum_element);
					/*if(data_index < 784 && datum_element != 0.0)
						printf("%d %f %hu, ", data_index, datum_element, dataNew[data_index]);*/
				}
			}
		}
		//printf("\n");

		char* dataNewBytes = (char*) malloc(size * sizeof(uint16_t));
		for (int j = 0; j < size * 2; j += 2) {
			dataNewBytes[j] = dataNew[j / 2] >> 8;
			dataNewBytes[j + 1] = dataNew[j / 2] & 0x00FF;
		}
		datumNew.set_data(dataNewBytes, size * 2);

		string output;
		datumNew.SerializeToString(&output);

		txn->Put(cursor_read->key(), output);

		free(dataNew);
		free(dataNewBytes);
		count++;
		cursor_read->Next();
	}
	txn->Commit();
	printf("\n%d\n\n", count);
}

int main(int argc, char** argv) {
	printf("Converting to posit(%d, %d)\n", _G_NBITS, _G_ESIZE);

	const string& mean_file = "/home/himeshi/dl/caffe-stripped/examples/cifar10/mean.binaryproto";

	ReadProtoFromBinaryFile(mean_file.c_str(), &data_mean);

	convert_train_db();

	convert_test_db();


	//Validation
/*	scoped_ptr<db::DB> db_read_test(db::GetDB (FLAGS_backend));
	db_read_test->Open("/home/himeshi/dl/caffe-stripped/examples/mnist/mnist_test_posit_lmdb", db::READ);
	scoped_ptr<db::Cursor> cursor_read_test(db_read_test->NewCursor());
	int newcount = 0;
	while(cursor_read_test->valid() && newcount < 2) {
		datumNew.ParseFromString(cursor_read_test->value());
		const int datum_channels_new = datumNew.channels();
		const int datum_height_new = datumNew.height();
		const int datum_width_new = datumNew.width();
		const string& dataNewdata = datumNew.data();
		fp16 datum_posit_element;
		for (int i = 0; i < datum_channels_new * datum_height_new * datum_width_new; i++) {
			datum_posit_element = static_cast<uint16_t>(static_cast<uint8_t>(dataNewdata[i * 2])) << 8 | static_cast<uint8_t>(dataNewdata[i * 2 + 1]);
			if(i < 784 && datum_posit_element != 0)
				printf("%d %hu ", i, (datum_posit_element));
		}
		cursor_read_test->Next();
		newcount++;
	}*/
}
