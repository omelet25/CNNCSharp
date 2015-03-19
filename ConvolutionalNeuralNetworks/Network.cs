using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using ConvolutionalNeuralNetworks.Layers;
using ConvolutionalNeuralNetworks.LossFunctions;

namespace ConvolutionalNeuralNetworks
{
	/// <summary>
	/// Network class
	/// </summary>
	/// <typeparam name="LossFunctionType">
	/// <para>Tools.LossFunction.MSE</para>
	/// <para>Tools.LossFunction.CrossEntropy</para>
	/// </typeparam>
	class Network<LossFunctionType> : IDisposable where LossFunctionType : ILossFunction, new()
	{
		private List<BaseLayer> _layers;
		private LossFunctionType _loss = new LossFunctionType();

		private Stopwatch stpw = new Stopwatch();

		/// <summary>
		/// <para>デフォルトコンストラクタ</para>
		/// </summary>
		public Network()
		{
			_layers = new List<BaseLayer>();
			LossFuncType = _loss.Type();
		}
		/// <summary>
		/// コンストラクタ
		/// </summary>
		/// <param name="layers">レイヤー</param>
		public Network(params BaseLayer[] layers)
		{
			_layers = new List<BaseLayer>(layers);
			LossFuncType = _loss.Type();
		}

		/// <summary>
		/// layerを1層追加
		/// </summary>
		/// <param name="layer"></param>
		public void AddLayer(BaseLayer layer) { _layers.Add(layer); }
		/// <summary>
		/// layerを複数層追加
		/// </summary>
		/// <param name="layers"></param>
		public void AddLayers(params BaseLayer[] layers) { _layers.AddRange(layers); }

		public double eta;
		public double mu;
		public double lambda;

		/// <summary>
		/// <para>学習</para>
		/// <para>batch_size >= train_input.Length or batch_size <= 0 : full-batch学習</para>
		/// <para>1 < batch_size < train_input.Length : mini-batch学習</para>
		/// <para>batch_size == 1 : online学習</para>
		/// <para>test_input と test_output が与えられた場合，毎回テストデータで誤差計算する</para>
		/// </summary>
		/// <param name="train_input">入力学習データ</param>
		/// <param name="train_teach">教師データ</param>
		/// <param name="test_input">テストデータ入力</param>
		/// <param name="test_output">テストデータ出力</param>
		/// <param name="batch_size">学習サンプルの与えるサイズ</param>
		/// <param name="epoch">学習回数</param>
		/// <param name="esp">許容誤差</param>
		/// <param name="learning_ratio">学習係数</param>
		/// <param name="dec_ratio">学習係数の減少率</param>
		/// <param name="momentum">モーメンタム係数</param>
		/// <param name="weight_decay">L2-正則化係数</param>
		/// <param name="each_epoch_call_back">各epochで呼ばれるコールバック</param>
		/// <param name="each_batch_call_back">各batchで呼ばれるコールバック</param>
		public void TrainWithTest(
			Vector<double>[] train_input, Vector<double>[] train_teach,
			Vector<double>[] test_input, Vector<double>[] test_output,
			int batch_size = 20, int epoch = 10000, double esp = 0.01,
			double learning_ratio = 0.05,double dec_ratio = 1.0, double momentum = 0.0, double weight_decay = 0.0,
			Func<bool> on_epoch = null,Func<double,bool> on_batch = null)
		{
			// イベントが null だとまずいので初期化
			if (on_epoch == null) { on_epoch = () => false; }
			if (on_batch == null) { on_batch = (val) => false; }

			eta = learning_ratio * Math.Sqrt(batch_size);
			mu = momentum;
			lambda = weight_decay;

			Console.WriteLine("Training start");
			stpw.Start();

			// Full-Batch にする
			if (batch_size > train_input.Length || batch_size <= 0) { batch_size = train_input.Length; }

			// disp count
			int _disp_cnt = train_input.Length / 50;

			for (int t = 0; t < epoch; t++)
			{
				// 学習データ誤差
				double _train_error = 0.0;

				//学習
				if (batch_size == 1)
				{
					// Online学習
					for (int i = 0; i < train_input.Length; i++)
					{
						if (i % _disp_cnt == 0) { Console.Write("*"); }
						_train_error = train_online(train_input[i], train_teach[i]);
						on_batch(_train_error);
					}
				}
				else if (batch_size == train_input.Length)
				{
					// Full-batch学習
					_train_error = train_batch(train_input, train_teach);
					on_batch(_train_error / batch_size);
				}
				else
				{
					// Mini-batch学習
					for (int i = 0; i < train_input.Length; i += batch_size)
					{
						if (i % _disp_cnt == 0) { Console.Write("*"); }
						int _bs = (i + batch_size) > train_input.Length ? (i + batch_size - train_input.Length) : batch_size;
						_train_error = train_batch(train_input.Skip(i).Take(_bs).ToArray(), train_teach.Skip(i).Take(_bs).ToArray());
						on_batch(_train_error / batch_size);
					}
				}

				Console.WriteLine();

				// テストデータでの実験
				if (on_epoch()) { break; }
			}

			Console.WriteLine("Elapsed Time : " + stpw.ElapsedMilliseconds / 1000.0 + " seconds");
		}

		//////////////////////////////////////////////////////////////////////////////////////////
		// Training 関連メソッド

		/// <summary>
		/// Online学習
		/// </summary>
		/// <param name="t_in">学習データ入力</param>
		/// <param name="t_te">教師データ</param>
		/// <returns></returns>
		private double train_online(Vector<double> t_in, Vector<double> t_te)
		{
			// テストデータの出力を計算
			var y = ForwardPropagation(t_in);

			// 出力層の誤差
			double _error = 0.0;
			
			// 出力層の誤差の微分 δ_o
			var _derr = y.Clone();
			for (int i = 0; i < _derr.Count; i++)
			{
				_error += _loss.f(y[i], t_te[i]);
				_derr[i] = _loss.df(_derr[i], t_te[i]);
			}

			// 誤差確認して許容誤差外だったら荷重更新
			for (int i = _layers.Count - 1; i >= 0; i--)
			{
				_derr = _layers[i].BackPropagation(_derr.Clone());
				// 荷重更新
				_layers[i].WeightUpdate(eta, mu, lambda);
			}

			return _error;
		}

		/// <summary>
		/// Batch or Mimi-batch学習
		/// </summary>
		/// <param name="t_in">学習データのブロック</param>
		/// <param name="t_te">教師データのブロック</param>
		/// <returns></returns>
		private double train_batch(Vector<double>[] t_in, Vector<double>[] t_te)
		{
			// 出力層の誤差
			double _error = 0.0;

			for (int i = 0; i < t_in.Length; i++)
			{
				// テストデータの出力を計算
				var y = ForwardPropagation(t_in[i]);

				// 出力層の誤差の微分 δ_o
				var _derr = y.Clone();
				for (int j = 0; j < _derr.Count; j++)
				{
					_error += _loss.f(y[j], t_te[i][j]);
					_derr[j] = _loss.df(_derr[j], t_te[i][j]);
				}
				// BP
				for (int j = _layers.Count - 1; j >= 0; j--)
				{
					_derr = _layers[j].BackPropagation(_derr.Clone());
				}
			}

			for (int i = 0; i < _layers.Count; i++) { _layers[i].WeightUpdate(eta, mu, lambda); }
		
			return _error;
		}

		/// <summary>
		/// 順方向計算(学習時使用)
		/// </summary>
		/// <param name="t_in">学習データ入力</param>
		/// <returns>出力</returns>
		private Vector<double> ForwardPropagation(Vector<double> t_in)
		{
			_layers[0].Inputs = t_in;
			for (int i = 0; i < _layers.Count; i++)
			{
				_layers[i].ForwardPropagation();
				if (i == _layers.Count - 1) { break; }
				_layers[i + 1].Inputs = _layers[i].Outputs;
			}
			return _layers.Last().Outputs;
		}

		//
		///////////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////////
		// 以下 Test関連メソッド

		/// <summary>
		/// <para>テスト(仮)</para>
		/// <para>入力データ，出力，誤差表示</para>
		/// </summary>
		/// <param name="t_input">入力</param>
		/// <param name="t_teach">教師</param>
		public Vector<double> Test(Vector<double> t_input, Vector<double> t_teach)
		{
			Console.WriteLine("Test start");

			double E = 0.0;
			var y = Prediction(t_input);
			for (int i = 0; i < y.Count; i++)
			{
				E += _loss.f(y[i], t_teach[i]);
			}
			Console.Write("Input:\t");
			foreach (var _ in t_input) { Console.Write(_ + " "); }
			Console.WriteLine();
			Console.Write("Output:\t");
			foreach (var _ in y) { Console.Write(_ + " "); }
			Console.WriteLine();
			Console.WriteLine("Error:\t" + E);

			return y.Clone();
		}

		/// <summary>
		/// 特定用途
		/// </summary>
		/// <param name="test_input"></param>
		/// <param name="test_output"></param>
		/// <returns></returns>
		private double Tests(Vector<double>[] test_input, Vector<double>[] test_output)
		{
			Console.WriteLine("Test start");

			int _acc = 0;
			int[,] recog_table = new int[10, 10];

			for (int i = 0; i < test_input.Length; i++)
			{
				var _output = Prediction(test_input[i]);
				var _om = _output.MaximumIndex();
				var _tom = test_output[i].MaximumIndex();

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

			Console.WriteLine("accuracy : " + _acc / (double)test_input.Length * 100.0);
			return _acc / (double)test_input.Length * 100.0;
		}


		/// <summary>
		/// 順方向計算(推定時使用)
		/// </summary>
		/// <param name="t_in"></param>
		/// <returns></returns>
		public Vector<double> Prediction(Vector<double> t_in)
		{
			_layers[0].Inputs = t_in;
			for (int i = 0; i < _layers.Count; i++)
			{
				_layers[i].ForwardPropagation();
				if (i == _layers.Count - 1) { break; }
				_layers[i + 1].Inputs = _layers[i].PredictOutputs;
			}
			return _layers.Last().PredictOutputs;
		}

		//
		//////////////////////////////////////////////////////////////////////////////////////////

		/// <summary>
		/// <para>ネットワークの構造が正しいかどうか</para>
		/// <para>入力と出力間のサイズ</para>
		/// </summary>
		/// <returns></returns>
		public bool NetworkCheck()
		{
			bool _flag = true;
			for (int i = 0; i < _layers.Count - 1; i++)
			{
				if (!_layers[i + 1].CheckSize(_layers[i].Outputs.Count))
				{
					if (_layers[i].LayerName == "" || _layers[i + 1].LayerName == "")
					{
						Console.WriteLine("Network structure is error." + i + " and " + (i + 1) + " layer.");
					}
					else
					{
						Console.WriteLine("Network structure is error." + _layers[i].LayerName + " and " + _layers[i + 1].LayerName + " layer.");
					}
					_flag = false;
				}
			}
			return _flag;
		}

		/// <summary>
		/// <para>ネットワークの構造出力</para>
		/// <para>[Layer No],[Layer Type],[Layer Option]</para>
		/// <para>Layer Option:ActivationType, PoolingType, ElementWizeType etc.</para>
		/// </summary>
		/// <returns></returns>
		public string NetworkStructure(string fmt = "d")
		{
			string structure = "";

			switch (fmt)
			{
				case "d":
					for (int i = 0; i < _layers.Count; i++)
					{
						structure += (i + 1) + ", " + _layers[i].LayerType + ", " + _layers[i].GenericsType + ", " + _layers[i].ToString("l") + "\n";
					}
					structure += (_layers.Count + 1) + ", OutputLayer, " + LossFuncType + "\n";
					break;
				default:
					for (int i = 0; i < _layers.Count; i++)
					{
						structure += (i + 1) + ", " + _layers[i].LayerType + "\n";
					}
					structure += (_layers.Count + 1) + ", OutputLayer" + "\n";
					break;
			}

			return structure;
		}

		public bool WriteWeights(string prefix = "")
		{
			foreach (var _l in _layers)
			{
				using (System.IO.StreamWriter sw = new System.IO.StreamWriter(prefix + "_weight_" + _l.LayerName + ".txt"))
				{
					sw.WriteLine(_l.ToString("w"));
				}
			}

			return true;
		}

		public bool WriteBiases(string prefix = "")
		{
			foreach (var _l in _layers)
			{
				using (System.IO.StreamWriter sw = new System.IO.StreamWriter(prefix + "_biase_" + _l.LayerName + ".txt"))
				{
					sw.WriteLine(_l.ToString("b"));
				}
			}

			return true;
		}

		public void Dispose() { _layers.Clear(); }

		public LossFunctionType LossFunction { get { return _loss; } }
		public string LossFuncType { get; private set; }
		public string TrainType { get; private set; }
	}
}
