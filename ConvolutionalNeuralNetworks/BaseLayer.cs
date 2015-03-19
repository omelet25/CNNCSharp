using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace ConvolutionalNeuralNetworks
{
	namespace Layers
	{
		/// <summary>
		/// Layerのbaseクラス
		/// </summary>
		abstract class BaseLayer
		{
			public BaseLayer(string layer_name)
			{
				LayerName = layer_name;
				LayerType = "BaseLayer";
				GenericsType = "None";

				// 使用しないパラメータは -1 とする
				_in_size = _out_size = _stride = -1;
			}

			/// <summary>
			/// <para>テスト(推定)時に使用する順方向計算</para>
			/// </summary>
			public virtual void Prediction() { ForwardPropagation(); }

			/// <summary>
			/// <para>順方向への計算</para>
			/// </summary>
			/// <returns></returns>
			public abstract void ForwardPropagation();

			/// <summary>
			/// 逆方向への計算
			/// </summary>
			/// <param name="next_delta">次層から来た誤差信号</param>
			public abstract Vector<double> BackPropagation(Vector<double> next_delta);

			/// <summary>
			/// 荷重，バイアスの更新
			/// </summary>
			/// <param name="eta">学習係数</param>
			/// <param name="mu">momentum係数</param>
			/// <param name="lambda">weight decay係数</param>
			public virtual void WeightUpdate(double eta, double mu, double lambda) { }

			/// <summary>
			/// 荷重，フィルタの生成．必要のないレイヤーでは何もしない
			/// </summary>
			public virtual void GenerateWeights(double lower = -0.1, double upper = 0.1) { }
			public virtual void GenerateWeights(Vector<double> weights) { }

			/// <summary>
			/// レイヤータイプ
			/// </summary>
			public string LayerType { get; protected set; }
			/// <summary>
			/// レイヤー名
			/// </summary>
			public string LayerName { get; protected set; }
			/// <summary>
			/// <para>子クラスが持つジェネリクスのType</para>
			/// <para>ActivationType or PoolingType or ElementWiseType</para>
			/// </summary>
			public string GenericsType { get; protected set; }

			/// <summary>
			/// 入力を Vector で set
			/// </summary>
			public abstract Vector<double> Inputs { set; }

			/// <summary>
			/// <para>出力を Vector で get (学習時のニューロン出力)</para>
			/// </summary>
			public abstract Vector<double> Outputs { get; }

			/// <summary>
			/// <para>出力を Vector で get (推定時のニューロン出力)</para>
			/// <para>Droped Layer以外ではOutputsと等しい</para>
			/// </summary>
			public virtual Vector<double> PredictOutputs { get { return Outputs; } }

			/// <summary>
			/// <para>荷重，フィルタを Vector で get,set</para>
			/// <para>持たない層では何もおこらない</para>
			/// </summary>
			public virtual Vector<double> Weights { get { return null; } protected set { } }

			/// <summary>
			/// <para>バイアスを Vector で get,set</para>
			/// </summary>
			public virtual Vector<double> Biases { get { return null; } protected set { } }

			/// <summary>
			/// Layer情報出力
			/// </summary>
			/// <param name="fmt">
			/// <para>"i" : 入力データ</para>
			/// <para>"p" : 入力データ(padding有)</para>
			/// <para>"w" : フィルタ or 荷重</para>
			/// <para>"o" : 出力データ</para>
			/// <para>"l" : レイヤー情報</para>
			/// <para>上記のfmt以外やレイヤーが持たない情報は"None"</para>
			/// </param>
			/// <returns>情報</returns>
			public virtual string ToString(string fmt) { return "None\n"; }
			
			/// <summary>
			/// <para>入力関連のパラメータ : _in_size</para>
			/// <para>size	: 総数</para>
			/// </summary>
			protected int _in_size;
			/// <summary>
			/// <para>出力関連のパラメータ : _out_size</para>
			/// <para>size	: 総数</para>
			/// </summary>
			protected int _out_size;
			/// <summary>
			/// stride
			/// </summary>
			protected int _stride;

			/// <summary>
			/// 前層の出力サイズと今の層の入力サイズを比較
			/// </summary>
			/// <param name="preLayerOutSize">前層からの出力</param>
			/// <returns></returns>
			public bool CheckSize(int preLayerOutSize) { return preLayerOutSize == _in_size; }
		}

		/// <summary>
		/// 局所コントラスト正規化層
		/// </summary>
		//class LclContNormLayer : Layer
		//{
		//}
	}
}
