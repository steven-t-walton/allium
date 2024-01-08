#include <iostream>

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
		return this->buff->sputc(c); 
	}
    int sync() {
    	write = 1; 
    	return this->buff->pubsync(); 
    }

	~CommentStreamBuf() { this->stream->rdbuf(this->buff); }
};

int main() {
	CommentStreamBuf buf(std::cout); 
	std::cout << "testing" << std::endl; 
	std::cout << "second" << std::endl; 
}