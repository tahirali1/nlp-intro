import tweepy
from tweepy import OAuthHandler
import nltk

# nltk.download()


class TwitterSentimentAnalysis(object):

    def __init__(self):
        self.consumer_key = 'AcHwTcdc8ndAq7tNVynt6BZkM'
        self.consumer_secret = 'sKasvpn6gLt6A5Ik5DO9dIBvSeULaDCBxlqA8JhyAh3K93V4rS'
        self.access_secret = 'qnasG2Rws3Ke41698rKIZ9agHwwHKidy4hGUv2vVsxbKi'
        self.access_token = '1089800802-XW0M6vXhfNO8V4aWcpQW3wz4I4ANBs0C241uhsq'

    def do_work(self):
        auth = OAuthHandler(consumer_key=self.consumer_key, consumer_secret=self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_secret)
        args = ['facebook']
        api = tweepy.API(auth, timeout=10)
        print(api)

        list_tweets = []
        query = args[0]
        if len(args) == 1:
            for status in tweepy.Cursor(api.search,
                                        q=query + '-filter:retweets',
                                        lang='en',
                                        result_type='recent',
                                        verify=False).items(100):
                list_tweets.append(status.text)
                print(status.text)

