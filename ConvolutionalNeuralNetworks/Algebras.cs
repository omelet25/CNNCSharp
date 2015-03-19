using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 多元数
namespace Algebras
{
	interface IAlgebraOperators<T>
	{
		/// <summary>
		/// 四則演算
		/// </summary>
		/// <param name="l_alg"></param>
		/// <param name="r_alg"></param>
		/// <returns></returns>
		T Add(T l_alg, T r_alg);
		T Sub(T l_alg, T r_alg);
		T Mul(T l_alg, T r_alg);
		T Mul(double lc, T r_alg);
		T Mul(T l_alg, double rc);
		T Div(T l_alg, T r_alg);
		T Div(double lc, T r_alg);
		T Div(T l_alg, double rc);
		/// <summary>
		/// 等号演算
		/// </summary>
		/// <param name="l_alg"></param>
		/// <param name="r_alg"></param>
		/// <returns></returns>
		bool Equals(T l_alg, T r_alg);
		bool NotEquals(T l_alg, T r_alg);

		/// <summary>
		/// 共役
		/// </summary>
		T Conjugate { get; }
		/// <summary>
		/// ノルム
		/// </summary>
		double Norm { get; }
		/// <summary>
		/// 逆元 1 / alg
		/// </summary>
		T Reciprocal { get; }
		/// <summary>
		/// <para>alg = alg_0*e_0 + ... + alg_i*e_i + ... </para>
		/// <para>alg_i == 0 for all i => true</para>
		/// </summary>
		bool IsZero { get; }
	}

	/// <summary>
	/// 複素数
	/// </summary>
	struct Complex : IFormattable, IEquatable<Complex>, IAlgebraOperators<Complex>
	{
		// element
		private double re, im;

		public Complex(double re = 0, double im = 0)
		{
			this.re = re; this.im = im;
		}
		public Complex(Complex z) { this = z; }

		public double Re { get { return re; } set { re = value; } }
		public double Im { get { return im; } set { im = value; } }
		public double this[int element]
		{
			get
			{
				switch (element) { case 0: return re; case 1: return im; default: throw new ArgumentOutOfRangeException(); }
			}
			set
			{
				switch (element) { case 0: re = value; break; case 1: im = value; break; default: throw new ArgumentOutOfRangeException(); }
			}
		}
		public double this[string element]
		{
			get
			{
				switch (element) { case "re": return re; case "im": return im; default: throw new ArgumentOutOfRangeException(); }
			}
			set
			{
				switch (element) { case "re": re = value; break; case "im": im = value; break; default: throw new ArgumentOutOfRangeException(); }
			}
		}

		// Operator
		// addition
		static public Complex operator +(Complex z1, Complex z2) { return new Complex(z1.re + z2.re, z1.im + z2.im); }
		// subtraction
		static public Complex operator -(Complex z1, Complex z2) { return new Complex(z1.re - z2.re, z1.im - z2.im); }
		// multiplication
		static public Complex operator *(Complex z1, Complex z2)
		{
			return new Complex(z1.re * z2.re - z1.im * z2.im, z1.re * z2.im + z1.im * z2.re);
		}
		static public Complex operator *(Complex z1, double c) { return new Complex(z1.re * c, z1.im * c); }
		static public Complex operator *(double c, Complex z2) { return new Complex(z2.re * c, z2.im * c); }
		// division
		static public Complex operator /(Complex z1, Complex z2)
		{
			if (z2.re == 0 && z2.im == 0) { throw new DivideByZeroException(); }
			return new Complex(z1 * z2.Conjugate / (z2.re * z2.re + z2.im * z2.im));
		}
		static public Complex operator /(Complex z1, double c)
		{
			if (c == 0) { throw new DivideByZeroException(); }
			return new Complex(z1.re / c, z1.im / c);
		}
		static public Complex operator /(double c, Complex z2)
		{
			if (z2.re == 0 && z2.im == 0) { throw new DivideByZeroException(); }
			return new Complex(c * z2.Conjugate / (z2.re * z2.re + z2.im * z2.im));
		}
		// equality
		static public bool operator ==(Complex z1, Complex z2) { return z1.re == z2.re && z1.im == z2.im; }
		static public bool operator !=(Complex z1, Complex z2) { return !z1.Equals(z2); }

		// method
		public double Arg { get { return Math.Atan2(this.im, this.re); } }
		public double Norm { get { return Math.Sqrt(this.re * this.re + this.im * this.im); } }
		public Complex Conjugate { get { return new Complex(this.re, -this.im); } }
		public bool IsZero { get { return this.re == 0 && this.im == 0; } }
		public Complex Reciprocal { get { return One / this; } }

		// static method
		// create
		public static Complex Zero { get { return new Complex(); } }
		public static Complex One { get { return new Complex(re: 1); } }
		public static Complex ImOne { get { return new Complex(im: 1); } }
		public static Complex FromPolarCoordinates(double r, double phase) { return new Complex(r * Math.Cos(phase), r * Math.Sin(phase)); }

		// override
		public bool Equals(Complex other) { return this.re == other.re && this.im == other.im; }
		public override bool Equals(object obj) { return base.Equals(obj); }
		public override int GetHashCode() { return base.GetHashCode(); }
		public string ToString(string format, IFormatProvider formatProvider) { return "(" + re + "," + im + ")"; }

		// interface
		public Complex Add(Complex l_alg, Complex r_alg) { return l_alg + r_alg; }
		public Complex Sub(Complex l_alg, Complex r_alg) { return l_alg - r_alg; }
		public Complex Mul(Complex l_alg, Complex r_alg) { return l_alg * r_alg; }
		public Complex Mul(double lc, Complex r_alg) { return lc * r_alg; }
		public Complex Mul(Complex l_alg, double rc) { return l_alg * rc; }
		public Complex Div(Complex l_alg, Complex r_alg) { return l_alg / r_alg; }
		public Complex Div(double lc, Complex r_alg) { return lc / r_alg; }
		public Complex Div(Complex l_alg, double rc) { return l_alg / rc; }
		public bool Equals(Complex l_alg, Complex r_alg) { return l_alg == r_alg; }
		public bool NotEquals(Complex l_alg, Complex r_alg) { return l_alg != r_alg; }
	}

	/// <summary>
	/// 四元数
	/// </summary>
	struct Quaternion : IFormattable, IEquatable<Quaternion>, IAlgebraOperators<Quaternion>
	{
		// element
		private double re, i, j, k;

		// constructor
		public Quaternion(double re = 0, double i = 0, double j = 0, double k = 0) { this.re = re; this.i = i; this.j = j; this.k = k; }
		public Quaternion(Quaternion q) { this = q; }

		// property
		public double Re { get { return re; } set { re = value; } }
		public double I { get { return i; } set { i = value; } }
		public double J { get { return j; } set { j = value; } }
		public double K { get { return k; } set { k = value; } }
		public double this[int element]
		{
			get
			{
				switch (element)
				{
					case 0: return re;
					case 1: return i;
					case 2: return j;
					case 3: return k;
					default: throw new ArgumentOutOfRangeException();
				}
			}
			set
			{
				switch (element)
				{
					case 0: re = value; break;
					case 1: i = value; break;
					case 2: j = value; break;
					case 3: k = value; break;
					default: throw new ArgumentOutOfRangeException();
				}
			}
		}
		public double this[string element]
		{
			get
			{
				switch (element)
				{
					case "re": return re;
					case "i": return i;
					case "j": return j;
					case "k": return k;
					default: throw new ArgumentOutOfRangeException();
				}
			}
			set
			{
				switch (element)
				{
					case "re": re = value; break;
					case "i": i = value; break;
					case "j": j = value; break;
					case "k": k = value; break;
					default: throw new ArgumentOutOfRangeException();
				}
			}
		}

		// Operator
		/// <summary>
		/// addition
		/// </summary>
		static public Quaternion operator +(Quaternion q1, Quaternion q2) { return new Quaternion(q1.re + q2.re, q1.i + q2.i, q1.j + q2.j, q1.k + q2.k); }
		/// <summary>
		/// subtraction
		/// </summary>
		static public Quaternion operator -(Quaternion q1, Quaternion q2) { return new Quaternion(q1.re - q2.re, q1.i - q2.i, q1.j - q2.j, q1.k - q2.k); }
		/// <summary>
		/// multiplication
		/// </summary>
		static public Quaternion operator *(Quaternion q1, Quaternion q2)
		{
			return new Quaternion(
				q1.re * q2.re - q1.i * q2.i - q1.j * q2.j - q1.k * q2.k,
				q1.re * q2.i + q1.i * q2.re + q1.j * q2.k - q1.k * q2.j,
				q1.re * q2.j - q1.i * q2.k + q1.j * q2.re + q1.k * q2.i,
				q1.re * q2.k + q1.i * q2.j - q1.j * q2.i + q1.k * q2.re);
		}
		static public Quaternion operator *(Quaternion q1, double c) { return new Quaternion(q1.re * c, q1.i * c, q1.j * c, q1.k * c); }
		static public Quaternion operator *(double c, Quaternion q2) { return q2 * c; }
		/// <summary>
		/// division
		/// </summary>
		static public Quaternion operator /(Quaternion q1, Quaternion q2)
		{
			if (q2.re == 0 && q2.i == 0 && q2.j == 0 && q2.k == 0) { throw new DivideByZeroException(); }
			return new Quaternion(q1 * q2.Conjugate / (q2.re * q2.re + q2.i * q2.i + q2.j * q2.j + q2.k * q2.k));
		}
		static public Quaternion operator /(Quaternion q1, double c)
		{
			if (c == 0) { throw new DivideByZeroException(); }
			return new Quaternion(q1.re / c, q1.i / c, q1.j / c, q1.k / c);
		}
		static public Quaternion operator /(double c, Quaternion q2)
		{
			if (q2.re == 0 && q2.i == 0 && q2.j == 0 && q2.k == 0) { throw new DivideByZeroException(); }
			return new Quaternion(c * q2.Conjugate / (q2.re * q2.re + q2.i * q2.i + q2.j * q2.j + q2.k * q2.k));
		}
		// equality
		static public bool operator ==(Quaternion lq, Quaternion rq) { return lq.re == rq.re && lq.i == rq.i && lq.j == rq.j && lq.k == rq.k; }
		static public bool operator !=(Quaternion lq, Quaternion rq) { return lq.re == rq.re || lq.i != rq.i || lq.j != rq.j || lq.k != rq.k; }

		// method
		public Quaternion Conjugate { get { return new Quaternion(this.re, -this.i, -this.j, -this.k); } }
		public Quaternion Reciprocal { get { return 1.0 / this; } }
		public double Norm { get { return Math.Sqrt(this.re * this.re + this.i * this.i + this.j * this.j + this.k * this.k); } }
		public bool IsZero { get { return this.re == 0 && this.i == 0 && this.j == 0 && this.k == 0; } }
		public bool IsPure { get { return this.re == 0; } }

		// static method
		// create
		public static Quaternion Zero { get { return new Quaternion(); } }
		public static Quaternion One { get { return new Quaternion(re: 1.0); } }
		public static Quaternion IOne { get { return new Quaternion(i: 1.0); } }
		public static Quaternion JOne { get { return new Quaternion(j: 1.0); } }
		public static Quaternion KOne { get { return new Quaternion(k: 1.0); } }
		public static Quaternion Pure(double i = 0, double j = 0, double k = 0) { return new Quaternion(0, i, j, k); }

		// override
		public bool Equals(Quaternion other) { return this.re == other.re && this.i == other.i && this.j == other.j && this.k == other.k; }
		public override bool Equals(object obj) { return base.Equals(obj); }
		public override int GetHashCode() { return base.GetHashCode(); }
		public string ToString(string format, IFormatProvider formatProvider) { return "(" + re + "," + i + "," + j + "," + k + ")"; }

		// interface
		public Quaternion Add(Quaternion l_alg, Quaternion r_alg) { return l_alg + r_alg; }
		public Quaternion Sub(Quaternion l_alg, Quaternion r_alg) { return l_alg - r_alg; }
		public Quaternion Mul(Quaternion l_alg, Quaternion r_alg) { return l_alg * r_alg; }
		public Quaternion Mul(double lc, Quaternion r_alg) { return lc * r_alg; }
		public Quaternion Mul(Quaternion l_alg, double rc) { return l_alg * rc; }
		public Quaternion Div(Quaternion l_alg, Quaternion r_alg) { return l_alg / r_alg; }
		public Quaternion Div(double lc, Quaternion r_alg) { return lc / r_alg; }
		public Quaternion Div(Quaternion l_alg, double rc) { return l_alg / rc; }
		public bool Equals(Quaternion l_alg, Quaternion r_alg) { return l_alg == r_alg; }
		public bool NotEquals(Quaternion l_alg, Quaternion r_alg) { return l_alg != r_alg; }
	}
}