#include <iostream>

// modify a given ostream to always print a comment character 
// at the beginning of every line 
// used to make third party libraries like MFEM create output 
// that is YAML-compliant 
class CommentStreamBuf : public std::streambuf {
private:
	std::ostream *stream; 
	std::streambuf *buff;
	int write = 1; 
	char comment; 
public:
	CommentStreamBuf(std::ostream &_stream, char _comment='#') 
		: stream(&_stream), buff(_stream.rdbuf(this)), comment(_comment)
	{

	}	

	int overflow(int c) {
		if (write) {
			this->buff->sputc(comment); 
			this->buff->sputc(' '); 
			write = 0; 
		}
		if (c=='\n') write = 1; 
		return this->buff->sputc(c); 
	}
    int sync() {
    	write = 1; 
    	return this->buff->pubsync(); 
    }

	~CommentStreamBuf() { this->stream->rdbuf(this->buff); }
};