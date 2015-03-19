using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using MathNet.Numerics.LinearAlgebra;

using ConvolutionalNeuralNetworks.Layers;

namespace ConvolutionalNeuralNetworks
{
	namespace Test
	{
		/// <summary>
		/// MNISTデータでテスト
		/// </summary>
		class MNIST
		{
			/// <summary>
			/// 学習データ：入力s
			/// </summary>
			private Vector<double>[] _train_inputs;
			/// <summary>
			/// 学習データ：出力
			/// </summary>
			private Vector<double>[] _train_outputs;
			/// <summary>
			/// テストデータ：入力
			/// </summary>
			private Vector<double>[] _test_inputs;
			/// <summary>
			/// テストデータ：出力
			/// </summary>
			private Vector<double>[] _test_outputs;

			private int _train_num;
			private int _test_num;

			private int _image_row;
			private int _image_col;

			private Network<LossFunctions.MSE> _LeNet;
			private Network<LossFunctions.MSE> _convnetjs;
			private Network<LossFunctions.MultiCrossEntropy> _mlp;

			public MNIST()
			{
				_train_num = 60000;
				_test_num = 10000;
				_image_row = 28;
				_image_col = 28;

				LoadMNISTImageDatas();
				LoadMNISTLabelDatas();

				_LeNet = new Network<LossFunctions.MSE>(
					new ConvolutionalLayer<Activations.Tanh>(28, 28, 1, 5, 6, 1, 2),
					new PoolingLayer<Poolings.Max>(28, 28, 6, 2, 2),
					new ConvolutionalLayer<Activations.Tanh>(14, 14, 6, 5, 16, 1, 0,
						new int[]{
							1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,
							1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,
							1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,
							0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,
							0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,
							0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1
						}),
					new PoolingLayer<Poolings.Max>(10, 10, 16, 2, 2),
					new ConvolutionalLayer<Activations.Tanh>(5,5,16,5,120),
					new FullyConnectedLayer<Activations.Tanh>(120, 84),
					new FullyConnectedLayer<Activations.Sigmoid>(84, 10)
				);

				_convnetjs = new Network<LossFunctions.MSE>(
					new ConvolutionalLayer<Activations.Sigmoid>(28, 28, 1, 5, 8, 1, 0),
					new PoolingLayer<Poolings.Max>(24, 24, 8, 2, 2),
					new ConvolutionalLayer<Activations.Sigmoid>(12, 12, 8, 5, 16, 1, 2),
					new PoolingLayer<Poolings.Max>(12, 12, 16, 3, 3),
					new FullyConnectedLayer<Activations.Sigmoid>(4 * 4 * 16, 10)
				);

				_mlp = new Network<LossFunctions.MultiCrossEntropy>(
					new FullyConnectedLayer<Activations.ReLU>(_image_row * _image_col, 32),
					new SoftmaxLayer(32, 10)
				);
			}

			/// <summary>
			/// 実行
			/// </summary>
			/// <param name="net_type">
			/// <para>network type</para>
			/// <para>net_type = 0 : MLP</para>
			/// <para>net_type = 1 : CNN</para>
			/// </param>
			public void Start(int net_type = 0)
			{
				switch (net_type)
				{
					case 0:
						Console.WriteLine(_mlp.NetworkStructure("d"));
						_mlp.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs, 1, 5,
							esp: 1, learning_ratio: 0.01, momentum: 0.0,weight_decay:0.0001,
							on_epoch: new Func<bool>(() => { _mlp.WriteWeights("_epoch_"); return true; }), on_batch: onBatch);
						break;
					case 1:
						Console.WriteLine(_LeNet.NetworkStructure("d"));
						if (!_LeNet.NetworkCheck()) { break; }
						_LeNet.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs, 1, 200, esp: 1, learning_ratio: 0.01);
						break;
					case 2:
						Console.WriteLine(_convnetjs.NetworkStructure("d"));
						if (!_convnetjs.NetworkCheck()) { break; }
						_convnetjs.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs,
							batch_size: 20,epoch: 100,esp:1,learning_ratio: 0.01,dec_ratio: 0.85,momentum: 0.5,weight_decay: 0.0001);
						break;
					default:
						break;
				}
			}

			private void LoadMNISTImageDatas()
			{
				// train data
				using (FileStream fs = new FileStream("MNIST\\train-images.idx3-ubyte", FileMode.Open, FileAccess.Read))
				{
					byte[] buf = new byte[16];
					fs.Read(buf, 0, 16);
					_train_inputs = new Vector<double>[_train_num];
					for (int i = 0; i < _train_num; i++)
					{
						double[] val = new double[_image_row * _image_col];
						for (int j = 0; j < _image_row * _image_col; j++)
						{
							val[j] = fs.ReadByte() / 255.0;
						}
						_train_inputs[i] = Vector<double>.Build.Dense(val);
					}
				}

				// train data
				using (FileStream fs = new FileStream("MNIST\\t10k-images.idx3-ubyte", FileMode.Open, FileAccess.Read))
				{
					byte[] buf = new byte[16];
					fs.Read(buf, 0, 16);
					_test_inputs = new Vector<double>[_test_num];
					for (int i = 0; i < _test_num; i++)
					{
						double[] val = new double[_image_row * _image_col];
						for (int j = 0; j < _image_row * _image_col; j++)
						{
							val[j] = fs.ReadByte() / 255.0;
						}
						_test_inputs[i] = Vector<double>.Build.Dense(val);
					}
				}
			}

			private void LoadMNISTLabelDatas()
			{
				// train labels
				using (FileStream fs = new FileStream("MNIST\\train-labels.idx1-ubyte", FileMode.Open, FileAccess.Read))
				{
					byte[] buf = new byte[8];
					fs.Read(buf, 0, 8);
					_train_outputs = new Vector<double>[_train_num];
					for (int i = 0; i < _train_num; i++)
					{
						var label = fs.ReadByte();
						_train_outputs[i] = Vector<double>.Build.Dense(10, new Func<int, double>(j => { return j == label ? 1 : 0; }));
					}
				}

				// test labels
				using (FileStream fs = new FileStream("MNIST\\t10k-labels.idx1-ubyte", FileMode.Open, FileAccess.Read))
				{
					byte[] buf = new byte[8];
					fs.Read(buf, 0, 8);
					_test_outputs = new Vector<double>[_test_num];
					for (int i = 0; i < _test_num; i++)
					{
						var label = fs.ReadByte();
						_test_outputs[i] = Vector<double>.Build.Dense(10, new Func<int, double>(j => { return j == label ? 1 : 0; }));
					}
				}
				
			}

			static int call_back_epochs = 0;
			static int call_back_batches = 0;
			static StreamWriter sw = new StreamWriter("MNIST_result.txt");

			/// <summary>
			/// 特定用途
			/// </summary>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <returns></returns>
			private bool onEpoch()
			{
				Console.WriteLine("Test start");

				int _acc = 0;
				int[,] recog_table = new int[10, 10];

				for (int i = 0; i < _test_inputs.Length; i++)
				{
					var _output = _mlp.Prediction(_test_inputs[i]);
					var _om = _output.MaximumIndex();
					var _tom = _test_outputs[i].MaximumIndex();

					recog_table[_om, _tom]++;
					if (_om == _tom) { _acc++; }
				}

				for (int i = 0; i < 10; i++)
				{
					for (int j = 0; j < 10; j++)
					{
						Console.Write(recog_table[i, j] + "\t");
					}
					Console.WriteLine();
				}

				Console.WriteLine("accuracy : " + _acc / (double)_test_inputs.Length * 100.0);

				sw.WriteLine("#accuracy");
				sw.WriteLine(call_back_epochs + "\t" + _acc / (double)_test_inputs.Length * 100.0);
				sw.WriteLine("#accuracy table");
				for (int i = 0; i < 10; i++)
				{
					for (int j = 0; j < 10; j++)
					{
						sw.Write(recog_table[i, j] + "\t");
					}
					sw.WriteLine();
				}

				call_back_epochs++;

				return false;
			}

			private bool onBatch(double err)
			{
				call_back_batches++;
				return true;
			}
		}

		class HUMAN_EXTRACTION
		{
			/// <summary>
			/// 学習データ：入力
			/// </summary>
			private Vector<double>[] _train_inputs;
			/// <summary>
			/// 学習データ：出力
			/// </summary>
			private Vector<double>[] _train_outputs;
			
			/// <summary>
			/// テストデータ：入力
			/// </summary>
			private Vector<double>[] _test_inputs;
			/// <summary>
			/// テストデータ：出力
			/// </summary>
			private Vector<double>[] _test_outputs;

			private int _train_num;
			private int _test_num;

			private int _image_row;
			private int _image_col;

			private Network<LossFunctions.MSE> _net;
			private Network<LossFunctions.MSE> _net_rgb;
			private Network<LossFunctions.MSE> _net_gray;

			int[] _cnct_tbl1_rgb = new int[]{
				1,1,1, 0,0,0, 0,0,0, 1,1,1, 0,0,0 ,1,1,1, 1,1,1,
				0,0,0, 1,1,1, 0,0,0, 1,1,1, 1,1,1 ,0,0,0, 1,1,1,
				0,0,0, 0,0,0, 1,1,1, 0,0,0, 1,1,1 ,1,1,1, 1,1,1
			};

			int[] _cnct_tbl2_rgb = new int[]{
				1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0,
				1,0,1, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0,
				1,0,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1,
				1,1,0, 0,0,0, 0,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,1,0, 1,1,0,
				1,1,0, 0,0,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1, 0,1,1,
				1,1,0, 0,0,0, 0,0,0, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1,0,1, 1,0,1,
				1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1,
			};

			int[] _cnct_tbl1_gray = new int[]{ 
				1,1,1,0,0,0,1,1,0,1,
				1,0,0,1,1,0,1,0,1,1,
				0,1,0,1,0,1,1,1,1,1,
				0,0,1,0,1,1,0,1,1,1
			};

			const string OUTPUT_PREFIX = "HUMAN_EXTRACTION\\";

			public HUMAN_EXTRACTION(int img_row,int img_col)
			{
				_image_row = img_row;
				_image_col = img_col;

				_net_rgb = new Network<LossFunctions.MSE>(
					new ConvolutionalLayer<Activations.Sigmoid>(_image_row, _image_col, 3, 9, 21, 1, 4, _cnct_tbl1_rgb, "C1"),
					new PoolingLayer<Poolings.Max>(96, 48, 21, 3, 3, "P2"),
					new ElemWiseLayer<ElementWises.MaxOut>(32, 16, 21, 3, 3, "E3"),
					new ConvolutionalLayer<Activations.Sigmoid>(32, 16, 7, 5, 33, 1, 2, _cnct_tbl2_rgb, "C4"),
					new PoolingLayer<Poolings.Max>(32, 16, 33, 2, 2, "P5"),
					new ElemWiseLayer<ElementWises.MaxOut>(16, 8, 33, 3, 3, "E6"),
					new ConvolutionalLayer<Activations.Sigmoid>(16, 8, 11, 5, 16, 1, 2, null, "C7"),
					new DropConnectLayer<Activations.Sigmoid>(16 * 8 * 16, 1024, 0.5, "DC8"),
					new FullyConnectedLayer<Activations.Sigmoid>(1024, _image_row * _image_col, "F9")
				);

				//Vector<double> _wei_C1 = null;
				//Vector<double> _wei_C4 = null;
				//Vector<double> _wei_DC6 = null;
				//Vector<double> _wei_F7 = null;
				//Vector<double> _bis_C1 = null;
				//Vector<double> _bis_C4 = null;
				//Vector<double> _bis_DC6 = null;
				//Vector<double> _bis_F7 = null;
				var _wei_C1 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_C1.txt");
				var _wei_C4 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_C4.txt");
				var _wei_DC6 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_DC6.txt");
				var _wei_F7 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_F7.txt");
				var _bis_C1 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_C1.txt");
				var _bis_C4 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_C4.txt");
				var _bis_DC6 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX +"result\\lr_0.05_biase_DC6.txt");
				var _bis_F7 = Tools.Utilities.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_F7.txt");


				_net_gray = new Network<LossFunctions.MSE>(
					new ConvolutionalLayer<Activations.Sigmoid>(_image_row, _image_col, 1, 9, 8, 1, 4, null, "C1", _wei_C1, _bis_C1),
					new PoolingLayer<Poolings.Max>(96, 48, 8, 3, 3, "P2"),
					new ElemWiseLayer<ElementWises.MaxOut>(32, 16, 8, 2, 2, "E3"),
					new ConvolutionalLayer<Activations.Sigmoid>(32, 16, 4, 5, 10, 1, 2, _cnct_tbl1_gray, "C4", _wei_C4, _bis_C4),
					new PoolingLayer<Poolings.Max>(32, 16, 10, 2, 2, "P5"),
					new DropConnectLayer<Activations.Sigmoid>(16 * 8 * 10, 1156, 0.5, "DC6", _wei_DC6, _bis_DC6),
					new FullyConnectedLayer<Activations.Sigmoid>(1156, _image_row * _image_col, "F7", _wei_F7, _bis_F7)
				);
			}

			public double _eta;

			int call_back_epochs = 0;
			int call_back_batches = 0;
			StreamWriter sw_epoch;
			StreamWriter sw_batch;

			double pre_error = double.MaxValue;

			public void Start(int net)
			{
				sw_epoch = new StreamWriter(OUTPUT_PREFIX + "result\\lr_"+_eta+"_on_epoch.txt");
				sw_epoch.WriteLine("#epoch\ttrain error\ttrain error per image\ttrain error per pix");
				sw_batch = new StreamWriter(OUTPUT_PREFIX + "result\\lr_"+_eta+"_on_batch.txt");

				Console.WriteLine("datas loading start");
				switch (net)
				{
					case 0:
						Tools.Utilities.LoadDataList(OUTPUT_PREFIX+"data\\train_4_in_RGB_wh.txt", out _train_inputs, out _train_outputs, true);
						Tools.Utilities.LoadDataList(OUTPUT_PREFIX+"data\\test_4_in_RGB_wh.txt", out _test_inputs, out _test_outputs, false);
						_net = _net_rgb;
						break;
					case 1:
						Tools.Utilities.LoadDataList(OUTPUT_PREFIX+"data\\train_567_in_L_wh_sc.txt", out _train_inputs, out _train_outputs, true);
						Tools.Utilities.LoadDataList(OUTPUT_PREFIX+"data\\test_567_in_L_wh_sc.txt", out _test_inputs, out _test_outputs, false);
						_net = _net_gray;
						break;
				}

				_train_num = _train_inputs.Length;
				_test_num = _test_inputs.Length;

				_net.NetworkStructure();
				if (!_net.NetworkCheck()) { return; }

				_net.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs,
					1, 20, 10, _eta, 1.0, 0.0, 0.001, onEpoch, onBatch);

				sw_epoch.Close();
				sw_batch.Close();
			}

			double tr_err = 0.0;

			private bool onEpoch()
			{
				Console.WriteLine("lr:" + _net.eta + "\tmoment:" + _net.mu + "\tweight decay" + _net.lambda);

				Console.WriteLine("Test start");
				Vector<double>[] _ys = new Vector<double>[_test_inputs.Length];

				// 誤差
				double ts_err = 0;
				for (int i = 0; i < _test_inputs.Length; i++)
				{
					_ys[i] = _net.Prediction(_test_inputs[i]);
					for (int j = 0; j < _ys[i].Count; j++)
					{
						ts_err += _net.LossFunction.f(_ys[i][j], _test_outputs[i][j]);
					}
				}
				ts_err = Math.Sqrt(ts_err);
				var ts_err_per_img = ts_err / _test_inputs.Length;
				var ts_err_per_pix = ts_err_per_img / (_image_row * _image_col);
				tr_err = Math.Sqrt(tr_err);
				var tr_err_per_img = tr_err / _train_inputs.Length;
				var tr_err_per_pix = tr_err_per_img / (_image_row * _image_col);
				
				Console.WriteLine(call_back_epochs + "\t" + tr_err + "\t" + tr_err_per_img + "\t" + tr_err_per_pix + "\t" + ts_err + "\t" + ts_err_per_img + "\t" + ts_err_per_pix);
				sw_epoch.WriteLine(call_back_epochs + "\t" + tr_err + "\t" + tr_err_per_img + "\t" + tr_err_per_pix + "\t" + ts_err + "\t" + ts_err_per_img + "\t" + ts_err_per_pix);

				if (ts_err < pre_error)
				{
					// 荷重出力
					_net.WriteWeights(OUTPUT_PREFIX + "result\\lr_" + _eta);
					// バイアス
					_net.WriteBiases(OUTPUT_PREFIX + "result\\lr_" + _eta);

					using (StreamWriter _sw = new StreamWriter(OUTPUT_PREFIX + "result\\lr_" + _eta + "_outputs.cnno"))
					{
						_sw.WriteLine(_image_row + "\t" + _image_col);
						for (int i = 0; i < _test_inputs.Length; i++)
						{
							foreach (var _v in _ys[i])
							{
								_sw.Write((_v * 255) + "\t");
							}
							_sw.WriteLine();
						}
					}
				}

				if (ts_err < pre_error) { pre_error = ts_err; }
				else
				{
					_net.eta = Math.Max(1e-5, _net.eta * 0.5);
					_net.mu = Math.Min(1.0 - 1.0e-5, _net.mu * 1.05);
				}

				call_back_epochs++;

				// train_dataを並び替え
				int[] _idx = Tools.Utilities.RandomIndex(0, _train_num, _train_num);
				Vector<double>[] _tmp_train_in = new Vector<double>[_train_num];
				Vector<double>[] _tmp_train_out = new Vector<double>[_train_num];
				for (int i = 0; i < _train_num; i++)
				{
					_tmp_train_in[i] = _train_inputs[_idx[i]].Clone();
					_tmp_train_out[i] = _train_outputs[_idx[i]].Clone();
				}
				_train_inputs = _tmp_train_in;
				_train_outputs = _tmp_train_out;

				sw_batch.Flush();
				sw_epoch.Flush();

				tr_err = 0.0;

				return ts_err < 10;
			}

			private bool onBatch(double err)
			{
				tr_err += err;
				sw_batch.WriteLine(err);
				call_back_batches++;
				return true;
			}
		}

		class HUMAN_DIRECTION
		{
			/// <summary>
			/// 学習データ：入力
			/// </summary>
			private Vector<double>[] _train_inputs;
			/// <summary>
			/// 学習データ：出力
			/// </summary>
			private Vector<double>[] _train_outputs;

			/// <summary>
			/// テストデータ：入力
			/// </summary>
			private Vector<double>[] _test_inputs;
			/// <summary>
			/// テストデータ：出力
			/// </summary>
			private Vector<double>[] _test_outputs;

			private int _train_num;
			private int _test_num;

			private int _image_row;
			private int _image_col;

			/// <summary>
			/// ネットワーク
			/// </summary>
			private Network<LossFunctions.MultiCrossEntropy> _net_cnn;
			private Network<LossFunctions.MultiCrossEntropy> _net_mlp;

			/// <summary>
			/// コネクションテーブル１
			/// </summary>
			int[] _cnct_tbl1_rgb = new int[]{
				1,1,1,1, 0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1,
				0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0, 1,1,1,1,
				0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1
			};

			/// <summary>
			/// コネクションテーブル２
			/// </summary>
			int[] _cnct_tbl2_gray = new int[]{
				1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0,
				1,0,1, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0,
				1,0,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1,
				1,1,0, 0,0,0, 0,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,1,0, 1,1,0,
				1,1,0, 0,0,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1, 0,1,1,
				1,1,0, 0,0,0, 0,0,0, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1,0,1, 1,0,1,
				1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1,
			};

			public HUMAN_DIRECTION(int img_row, int img_col)
			{
				_image_row = img_row;
				_image_col = img_col;

				Console.WriteLine("datas loading start");
				Tools.Utilities.LoadDataList("HUMAN_DIRECTION\\train_all_in_wh.txt", out _train_inputs, out _train_outputs, true);
				Tools.Utilities.LoadDataList("HUMAN_DIRECTION\\test_all_in_wh.txt", out _test_inputs, out _test_outputs, true);

				_train_num = _train_inputs.Length;
				_test_num = _test_inputs.Length;

				var _image_dep = _train_inputs[0].Count / (_image_row * _image_col);

				_net_cnn = new Network<LossFunctions.MultiCrossEntropy>(
					new ConvolutionalLayer<Activations.Sigmoid>(_image_row, _image_col, _image_dep, 9, 28, 1, 4, _cnct_tbl1_rgb, "C1"),
					new PoolingLayer<Poolings.Max>(96, 48, 28, 2, 2, "P2"),
					new ElemWiseLayer<ElementWises.MaxOut>(48, 24, 28, 4, 4, "E3"),
					new ConvolutionalLayer<Activations.Sigmoid>(48, 24, 7, 5, 33, 1, 2, _cnct_tbl2_gray, "C4"),
					new PoolingLayer<Poolings.Max>(48, 24, 33, 2, 2, "P5"),
					new ElemWiseLayer<ElementWises.MaxOut>(24, 12, 33, 3, 3, "E6"),
					new ConvolutionalLayer<Activations.Sigmoid>(24, 12, 11, 5, 32, 1, 2, null, "C7"),
					new PoolingLayer<Poolings.Max>(24, 12, 32, 2, 2, "P8"),
					new DropOutLayer<Activations.Sigmoid>(12 * 6 * 32, 600, 0.5, "DO9"),
					new FullyConnectedLayer<Activations.Sigmoid>(600, 100, "F10"),
					new SoftmaxLayer(100, 8, "S11")
				);

				_net_mlp = new Network<LossFunctions.MultiCrossEntropy>(
					new DropConnectLayer<Activations.Sigmoid>(_image_row * _image_col * _image_dep, 16 * 8 * 3, 0.5, "L1"),
					new SoftmaxLayer(16 * 8 * 3, 8, "L2")
				);
			}

			Network<LossFunctions.MultiCrossEntropy> _net;

			static int call_back_epochs = 0;
			static int call_back_batches = 0;
			static StreamWriter sw_trerr = new StreamWriter("DIRECTION_train_err.txt");
			static StreamWriter sw_tserr = new StreamWriter("DIRECTION_test_err.txt");
			static StreamWriter sw_netlr = new StreamWriter("DIRECTION_net_lr.txt");
			double pre_acc = 0;
			double tr_err = 0;

			/// <summary>
			/// 特定用途
			/// </summary>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <returns></returns>
			private bool onEpoch()
			{
				Console.WriteLine(tr_err / _train_num);
				tr_err = 0;
				Console.WriteLine("Test start in epoch" + call_back_epochs);
				double _acc = 0;
				int[,] recog_table = new int[8, 8];
				for (int i = 0; i < _test_inputs.Length; i++)
				{
					var _output = _net.Prediction(_test_inputs[i]);
					var _om = _output.MaximumIndex();
					var _tom = _test_outputs[i].MaximumIndex();

					recog_table[_om, _tom]++;
					if (_om == _tom) { _acc++; }
				}
				// table表示
				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						Console.Write(recog_table[i, j] + "\t");
					}
					Console.WriteLine();
				}
				// 精度
				_acc = _acc / (double)_test_inputs.Length;
				Console.WriteLine("accuracy : " + _acc);
				sw_tserr.Write(call_back_epochs + "\t" + _acc);
				double each_ts_num = (_test_inputs.Length / (double)recog_table.GetLength(0));
				for (int i = 0; i < 8; i++)
				{
					sw_tserr.Write("\t" + (recog_table[i, i] / each_ts_num));
				}
				sw_tserr.WriteLine();

				// networkのパラメータ
				Console.WriteLine("lr:" + _net.eta + "\tmoment:" + _net.mu + "\tweight decay" + _net.lambda);
				sw_netlr.WriteLine(_net.eta + "\t" + _net.mu + "\t" + _net.lambda);

				call_back_epochs++;

				// train_dataを並び替え
				int[] _idx = Tools.Utilities.RandomIndex(0, _train_num, _train_num);
				Vector<double>[] _tmp_train_in = new Vector<double>[_train_num];
				Vector<double>[] _tmp_train_out = new Vector<double>[_train_num];
				for (int i = 0; i < _train_num; i++)
				{
					_tmp_train_in[i] = _train_inputs[_idx[i]].Clone();
					_tmp_train_out[i] = _train_outputs[_idx[i]].Clone();
				}
				_train_inputs = _tmp_train_in;
				_train_outputs = _tmp_train_out;

				// パラメータの調整
				if (pre_acc >= _acc)
				{
					_net.eta = Math.Max(1e-5, _net.eta * 0.5);
					_net.mu = Math.Min(1.0 - 1.0e-5, _net.mu * 1.5);
				}
				else { pre_acc = _acc; }

				// バッファクリア
				// 書き込み
				sw_trerr.Flush();
				sw_tserr.Flush();
				sw_netlr.Flush();

				if (_acc > 99.0) { return true; }
				else { return false; }
			}

			private bool onBatch(double err)
			{
				tr_err += err;
				sw_trerr.WriteLine(err);
				call_back_batches++;
				return true;
			}

			public void Start(int net_type = 0)
			{
				switch (net_type)
				{
					case 0:
						_net = _net_mlp;
						Console.WriteLine(_net.NetworkStructure("d"));
						if (!_net.NetworkCheck()) { break; }
						_net.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs, 1, epoch: 5000,
							esp: 1, learning_ratio: 0.05, momentum: 0.0, weight_decay: 0.001,
							on_epoch: onEpoch, on_batch: onBatch);
						break;
					case 1:
						_net = _net_cnn;
						Console.WriteLine(_net.NetworkStructure("d"));
						if (!_net.NetworkCheck()) { break; }
						_net.TrainWithTest(_train_inputs, _train_outputs, _test_inputs, _test_outputs, 5, 20,
							esp: 1, learning_ratio: 0.05, momentum: 0.0, weight_decay: 0.0001,
							on_epoch: onEpoch, on_batch: onBatch);
						break;
					default:
						break;
				}
			}
		}
	}
}
