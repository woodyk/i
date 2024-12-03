#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: data_sample.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 23:50:42
# Modified: 2024-12-03 00:34:13

SAMPLE_DATA = """
    The server at 192.168.1.1/24 manages local network traffic, while the router gateway is set to 10.0.0.1. A public web server is accessible at 203.0.113.45, and an alternative testing server uses 198.51.100.27/32. For internal systems, we use a small subnet like 172.16.0.0/16. Occasionally, a device might have a static IP of 192.0.2.10, and legacy systems still refer to older IPs like 127.0.0.1 for loopback or 8.8.8.8 for Google's DNS. A customer mentioned their IP being 169.254.1.5, which falls under the link-local range. Lastly, our firewall monitors traffic from 123.123.123.123, a public IP on a different network.

    2001:db8:3333:4444:5555:6666:7777:8888
    2001:db8:3333:4444:CCCC:DDDD:EEEE:FFFF
    :: (implies all 8 segments are zero)
    2001:db8:: (implies that the last six segments are zero)
    ::1234:5678
    ::2
    ::0bff:db8:1:1
       ::0bff:db8:1:1
    2001:db8::ff34:5678
    2001:db8::1:5678
    2001:0db8:0001:0000:0000:0ab9:C0A8:0102 (This can be compressed to eliminate leading zeros, as follows: 2001:db8:1::ab9:C0A8:102 )
    Send funds to support@example.org
       AA:BB:CC:DD:EE:FF
    aa:bb:cc:dd:ee:ff
    Aa:Bb:Cc:Dd:Ee:Ff
    AABBCCDDEEFF
        aabbccddeeff
    AaBbCcDdEeFf
    AA-BB-CC-DD-EE-FF
    aa-bb-cc-dd-ee-ff
    Aa-Bb-Cc-Dd-Ee-Ff
    AA.BB.CC.DD.EE.FF
    aa.bb.cc.dd.ee.ff
    Aa.Bb.Cc.Dd.Ee.Ff
    IPv6: fe80::1ff:fe23:4567:890a
    IPv6: fe80::1ff:fe23:4567:890a
    Give me a call at 1234567890
    Visit https://example.com for more info.
    Check the config at /etc/config/settings.ini or C:\\Windows\\System32\\drivers\\etc\\hosts.
    My phone number is 942 282 1445 or 954 224-3454 or (282) 445-4983
    +1 (203) 553-3294 and this 1-849-933-9938
    One Apr 4th 1922 at 12pm or 12:30 pm or 10:20am
    Here is some JSON: {"key": "value"} or an array: [1, 2, 3].
    IPv4: 192.168.1.1, IPv6: fe80::1ff:fe23:4567:890a, MAC: 00:1A:2B:3C:4D:5E.
    Timestamp: 2023-11-18T12:34:56Z, Hex: 0x1A2B3C, Env: $HOME or %APPDATA%.
    UUID: 550e8400-e29b-41d4-a716-446655440000
     https://localhost/test.html
       July 23rd 2023
    ssh://localhost:808/test
    11/19/2024 01:21:23
     11/19/2024 01:21:23
       2024/8/29
    12.03.24
    Jan March May July dec
    mon monday tues fri sunday
    IPv6: fe80::1ff:fe23:4567:890a
    Short: fe80::
    Dual: 2001:db8::192.168.1.1
    Invalid: 123::abc::456
    """

VALIDATE_DATA = """
Here are examples of various data types:

1. **IP Addresses**:
   IPv4: 192.168.1.1, 255.255.255.255, 10.0.0.1, 192.168.0.0/16, 127.0.0.1
   IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334, fe80::1, 2001:db8::/48

2. **URLs**:
   http://example.com, https://secure-site.org, ftp://ftp.example.com, mailto:user@example.com

3. **Email addresses**:
   user@example.com, another.user@domain.org, test123+filter@sub.domain.com

4. **Phone numbers**:
   +1 (123) 456-7890, 123-456-7890, 5555555555, +44 20 7946 0958, +91-9876543210

5. **Dates and times**:
   ISO date: 2023-09-01, 2023/09/01
   US date: 09/01/2023, 09-01-2023
   European date: 01/09/2023, 01-09-2023
   Date with month: 15 March 2023, 7 July 2021
   Times: 14:30, 09:45:30, 3:45 PM, 12:00 AM

6. **Postal codes**:
   USA: 12345, 90210-1234
   Canada: A1A 1A1, B2B-2B2
   UK: SW1A 1AA, EC1A 1BB
   Germany: 10115, France: 75008, Australia: 2000

7. **VIN numbers**:
   1HGCM82633A123456, JH4KA4650MC000000, 5YJSA1CN5DFP01234

8. **MAC addresses**:
   00:1A:2B:3C:4D:5E, 00-1A-2B-3C-4D-5E, 001A.2B3C.4D5E

9. **Routing numbers**:
   011000015, 121000358, 123456789 (Invalid), 021000021

10. **Bank account numbers**:
    IBAN: DE44 5001 0517 5407 3249 31, GB29 NWBK 6016 1331 9268 19
    US: 12345678901234567, 987654321

11. **Credit card numbers**:
    4111 1111 1111 1111 (Visa), 5500-0000-0000-0004 (MasterCard), 378282246310005 (American Express), 6011 1111 1111 1117 (Discover)

12. **Social Security Numbers (SSNs)**:
    123-45-6789, 987654321, 000-12-3456 (Invalid)

13. **Passport numbers**:
    USA: 123456789, UK: 987654321, Canada: A1234567, India: B9876543

14. **SWIFT codes**:
    BOFAUS3NXXX, CHASUS33, DEUTDEFF500, HSBCGB2LXXX

15. **Geo-coordinates**:
    Decimal: 37.7749, -122.4194; 40.7128, -74.0060
    Degrees, minutes, seconds: 40°42'51"N 74°00'21"W, 37°46'29"N 122°25'09"W

1HGCM82633A004352, JH4TB2H26CC000000, 5YJSA1CN5DFP01234
"""


